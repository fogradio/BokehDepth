#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical scatter renderer with an adjustable polygonal aperture (BokehMe).

Same idea as scatter.py, but the kernel parameterises the aperture as a
polygon with ``poly_sides`` vertices, allowing non-circular bokeh shapes.
"""

import re

import cupy
import torch

kernel_Render_updateOutput = '''

    extern "C" __global__ void kernel_Render_updateOutput(
        const int n,
        const int polySides,
        const float initAngle,
        const float* image,          // original image
        const float* defocus,        // signed defocus map
        int* defocusDilate,          // signed defocus map after dilating
        float* bokehCum,             // cumulative bokeh image
        float* weightCum             // cumulative weight map
    )
    {
        float PI = 3.1415926536;
        float fltAngle1 = 2 * PI / (float)(polySides);
        float fltAngle2 = PI / 2 - PI / (float)(polySides);
        float donutRatio = 0;  // (0 -> 0.5 : circle -> donut)

        for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int intN = ( intIndex / SIZE_3(weightCum) / SIZE_2(weightCum) / SIZE_1(weightCum) ) % SIZE_0(weightCum);
            const int intY = ( intIndex / SIZE_3(weightCum)                                         ) % SIZE_2(weightCum);
            const int intX = ( intIndex                                                             ) % SIZE_3(weightCum);

            float fltDefocus = VALUE_4(defocus, intN, 0, intY, intX);
            float fltRadius = fabsf(fltDefocus);

            for (int intDeltaY = -(int)(fltRadius)-1; intDeltaY <= (int)(fltRadius)+1; intDeltaY++) {
                for (int intDeltaX = -(int)(fltRadius)-1; intDeltaX <= (int)(fltRadius)+1; intDeltaX++) {

                    int intNeighborY = intY + intDeltaY;
                    int intNeighborX = intX + intDeltaX;

                    float fltAngle = atan2f((float)(intDeltaY), (float)(intDeltaX));
                    fltAngle = fmodf(fabsf(fltAngle + initAngle), fltAngle1);

                    if ((intNeighborY >= 0) & (intNeighborY < SIZE_2(bokehCum)) & (intNeighborX >= 0) & (intNeighborX < SIZE_3(bokehCum))) {
                        float fltDist = sqrtf((float)(intDeltaY)*(float)(intDeltaY) + (float)(intDeltaX)*(float)(intDeltaX));
                        float fltWeight = (0.5 + 0.5 * tanhf(4 * (fltRadius * sinf(fltAngle2)/sinf(fltAngle+fltAngle2) - fltDist))) * (1 - donutRatio + donutRatio * tanhf(0.2 * (1 + fltDist - fltRadius * sinf(fltAngle2)/sinf(fltAngle+fltAngle2)))) / (fltRadius * fltRadius + 0.2);
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
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


class _FunctionRender(torch.autograd.Function):
    @staticmethod
    def forward(self, image, defocus, poly_sides, init_angle):
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
                        "poly_sides": poly_sides,
                        "init_angle": init_angle,
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
                    cupy.int(n),
                    cupy.int(poly_sides),
                    cupy.float32(init_angle),
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


def FunctionRender(image, defocus, poly_sides, init_angle):
    return _FunctionRender.apply(image, defocus, poly_sides, init_angle)


class ModuleRenderScatterEX(torch.nn.Module):
    """Soft-disk scatter renderer with an adjustable polygonal aperture."""

    def forward(self, image, defocus, poly_sides=10000, init_angle=3.1415926536 / 2):
        defocus_dilate, bokeh_cum, weight_cum = FunctionRender(image, defocus, poly_sides, init_angle)
        bokeh = bokeh_cum / weight_cum
        return bokeh, defocus_dilate
