#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical scatter renderer with a circular aperture (BokehMe).

Ported from the BokehMe reference implementation. The CUDA kernel is launched
through CuPy and performs the soft-disk weighted scatter described in the
BokehMe paper.
"""

import re

import cupy
import numpy
import torch
import torch.nn as nn  # noqa: F401  (kept for downstream subclassing convenience)
import torch.nn.functional as F  # noqa: F401

kernel_Render_updateOutput = '''

    extern "C" __global__ void kernel_Render_updateOutput(
        const int n,
        const float* image,          // original image
        const float* defocus,        // signed defocus map
        int* defocusDilate,          // signed defocus map after dilating
        float* bokehCum,             // cumulative bokeh image
        float* weightCum             // cumulative weight map
    )
    {
        for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int intN = ( intIndex / SIZE_3(weightCum) / SIZE_2(weightCum) / SIZE_1(weightCum) ) % SIZE_0(weightCum);
            const int intY = ( intIndex / SIZE_3(weightCum)                                         ) % SIZE_2(weightCum);
            const int intX = ( intIndex                                                             ) % SIZE_3(weightCum);

            float fltDefocus = VALUE_4(defocus, intN, 0, intY, intX);
            float fltRadius = fabsf(fltDefocus);

            for (int intDeltaY = -(int)(fltRadius)-1; intDeltaY <= (int)(fltRadius)+1; ++intDeltaY) {
                for (int intDeltaX = -(int)(fltRadius)-1; intDeltaX <= (int)(fltRadius)+1; ++intDeltaX) {

                    int intNeighborY = intY + intDeltaY;
                    int intNeighborX = intX + intDeltaX;

                    if ((intNeighborY >= 0) && (intNeighborY < SIZE_2(bokehCum)) && (intNeighborX >= 0) && (intNeighborX < SIZE_3(bokehCum))) {
                        float fltDist = sqrtf((float)(intDeltaY)*(float)(intDeltaY) + (float)(intDeltaX)*(float)(intDeltaX));
                        float fltWeight = (0.5 + 0.5 * tanhf(4 * (fltRadius - fltDist))) / (fltRadius * fltRadius + 0.2);
                        if (fltRadius >= fltDist) {
                            atomicMax(&defocusDilate[OFFSET_4(defocusDilate, intN, 0, intNeighborY, intNeighborX)], int(fltDefocus));
                        }
                        atomicAdd(&weightCum[OFFSET_4(weightCum, intN, 0, intNeighborY, intNeighborX)], fltWeight);
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, intN, 0, intNeighborY, intNeighborX)], fltWeight * VALUE_4(image, intN, 0, intY, intX));
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, intN, 1, intNeighborY, intNeighborX)], fltWeight * VALUE_4(image, intN, 1, intY, intX));
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, intN, 2, intNeighborY, intNeighborX)], fltWeight * VALUE_4(image, intN, 2, intY, intX));
                    }
                }
            }
        }
    }

'''


def cupy_kernel(strFunction, objVariables):
    """Replace SIZE_/OFFSET_/VALUE_ placeholders in the CUDA template."""
    strKernel = globals()[strFunction]

    while True:
        objMatch = re.search(r"(SIZE_)([0-4])(\()([^\)]*)(\))", strKernel)
        if objMatch is None:
            break
        intArg = int(objMatch.group(2))
        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()
        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))

    while True:
        objMatch = re.search(r"(OFFSET_)([0-4])(\()([^\)]+)(\))", strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(",")
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [
            "((" + strArgs[intArg + 1].replace("{", "(").replace("}", ")").strip() + ")*" + str(intStrides[intArg]) + ")"
            for intArg in range(intArgs)
        ]
        strKernel = strKernel.replace(objMatch.group(0), "(" + str.join("+", strIndex) + ")")

    while True:
        objMatch = re.search(r"(VALUE_)([0-4])(\()([^\)]+)(\))", strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(",")
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [
            "((" + strArgs[intArg + 1].replace("{", "(").replace("}", ")").strip() + ")*" + str(intStrides[intArg]) + ")"
            for intArg in range(intArgs)
        ]
        strKernel = strKernel.replace(objMatch.group(0), strTensor + "[" + str.join("+", strIndex) + "]")

    return strKernel


@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    import cupy as cp

    return cp.RawKernel(strKernel, strFunction)


class _FunctionRender(torch.autograd.Function):
    @staticmethod
    def forward(self, image, defocus):
        defocus_dilate = defocus.int()
        bokeh_cum = torch.zeros_like(image)
        weight_cum = torch.zeros_like(defocus)

        if defocus.is_cuda:
            n = weight_cum.nelement()
            cupy_launch(
                "kernel_Render_updateOutput",
                cupy_kernel(
                    "kernel_Render_updateOutput",
                    {
                        "image": image,
                        "defocus": defocus,
                        "defocusDilate": defocus_dilate,
                        "bokehCum": bokeh_cum,
                        "weightCum": weight_cum,
                    },
                ),
            )(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    numpy.int32(n),
                    image.data_ptr(),
                    defocus.data_ptr(),
                    defocus_dilate.data_ptr(),
                    bokeh_cum.data_ptr(),
                    weight_cum.data_ptr(),
                ],
            )
        else:
            raise NotImplementedError("CPU scatter renderer is not implemented")

        return defocus_dilate.float(), bokeh_cum, weight_cum


def FunctionRender(image, defocus):
    defocus_dilate, bokeh_cum, weight_cum = _FunctionRender.apply(image, defocus)
    return defocus_dilate, bokeh_cum, weight_cum


class ModuleRenderScatter(nn.Module):
    """Soft-disk scatter renderer with a circular aperture."""

    def forward(self, image, defocus):
        defocus_dilate, bokeh_cum, weight_cum = FunctionRender(image, defocus)
        bokeh = bokeh_cum / weight_cum
        return bokeh, defocus_dilate
