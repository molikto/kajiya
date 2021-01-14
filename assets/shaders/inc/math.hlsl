#ifndef MATH_HLSL
#define MATH_HLSL

#include "math_const.hlsl"

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
float3x3 build_orthonormal_basis(float3 n) {
    float3 b1;
    float3 b2;

    if (n.z < 0.0) {
        const float a = 1.0 / (1.0 - n.z);
        const float b = n.x * n.y * a;
        b1 = float3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = float3(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        const float a = 1.0 / (1.0 + n.z);
        const float b = -n.x * n.y * a;
        b1 = float3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return float3x3(
        b1.x, b2.x, n.x,
        b1.y, b2.y, n.y,
        b1.z, b2.z, n.z
    );
}

#endif
