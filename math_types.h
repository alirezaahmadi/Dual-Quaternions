#pragma once

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

namespace DynaMap{
namespace math{

struct EulerAngles
{
    float roll, pitch, yaw;
};

struct AxisAngle
{
    float theta;
    float3 n;
};

}
}