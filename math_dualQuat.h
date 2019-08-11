#pragma once

#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "math/math_Quat.h"

namespace DynaMap{
namespace math{

class dualQuat {
    public:
    Quaternion real;
    Quaternion dual;

    __host__ __device__
    inline dualQuat(void){
        real = Quaternion(1,0,0,0);
        dual = Quaternion(0,0,0,0);
    }
    __host__ __device__
    inline dualQuat(Quaternion r, Quaternion d): real(r), dual(d){
        
    }
    __host__ __device__
    inline dualQuat(Quaternion r, float3 t){   // todo ... check the equation
        r = r.normalize();
        float w  = -0.5f*( t.x * r.x + t.y * r.y + t.z * r.z);
        float xi =  0.5f*( t.x * r.w + t.y * r.z - t.z * r.y);
        float yj =  0.5f*(-t.x * r.z + t.y * r.w + t.z * r.x);
        float zk =  0.5f*( t.x * r.y - t.y * r.x + t.z * r.w);
        // dualQuat(r, Quaternion(w, xi, yj, zk));
        real = r;
        dual = Quaternion(w, xi, yj, zk);
        //  ( new Quaternion( 0, t ) * real ) * 0.5f;
    }
    __host__ __device__
    inline dualQuat(float4x4& t){
        dualQuat dq(Quaternion(t.getFloat3x3()), make_float3(t.m14, t.m24, t.m34));
        real = dq.real;
        dual = dq.dual;
    }
    __host__ __device__
    inline static dualQuat identity(void){
        return dualQuat(Quaternion(1.0f, 0.0f, 0.0f, 0.0f), Quaternion(0.0f, 0.0f, 0.0f, 0.0f));
    }
    __host__ __device__
    inline void setIdentity(void){
        dualQuat dq(Quaternion(1.0f, 0.0f, 0.0f, 0.0f), Quaternion(0.0f, 0.0f, 0.0f, 0.0f));
        real = dq.real;
        dual = dq.dual;
    }
    /****************************************************************/
    /**************************Products******************************/
    /****************************************************************/
    __host__ __device__
    inline static dualQuat normalize(dualQuat q){
        float norm = q.real.norm();
        dualQuat dq;
        dq.real = q.real / norm;
        dq.dual = q.dual / norm;
        return dq;
    }
    __host__ __device__
    inline dualQuat normalize(void){
        float norm = real.norm();
        real /= norm;
        dual /= norm;
        return *this;
    }
    __host__ __device__
    inline static float dot(dualQuat a, dualQuat b){
        return Quaternion::dot(a.real, b.real);
    }
    __host__ __device__
    inline float dot(dualQuat b){
        return Quaternion::dot(real, b.real);
    }
    __host__ __device__ 
    inline static dualQuat mul(dualQuat a, dualQuat b){
        return dualQuat(a.real * b.real, (a.real* b.dual) + (b.real * a.dual));
    }
    __host__ __device__ 
    inline dualQuat mul(dualQuat b){
        return dualQuat(real * b.real, (real* b.dual) + (b.real * dual));
    }
    __host__ __device__ 
    inline static dualQuat conjugate(dualQuat q){
        return dualQuat(Quaternion::conjugate(q.real), Quaternion::conjugate(q.dual));
    }
    __host__ __device__ 
    inline dualQuat conjugate(void){
        return dualQuat(Quaternion::conjugate(real), Quaternion::conjugate(dual));
    }
    __host__ __device__ 
    inline static dualQuat inverse(dualQuat q){
        assert(q.real != 0);
        return dualQuat(q.real.inverse() ,  (q.real.inverse() - (q.real.inverse() * q.dual * q.real.inverse()))); 
    }
    __host__ __device__ 
    inline dualQuat inverse(void){
        assert(real != 0);
        return dualQuat(real.inverse() , (real.inverse() - (real.inverse() * dual * real.inverse())));
    } 
    /****************************************************************/
    /**************************Operators*****************************/
    /****************************************************************/
    __host__ __device__
    inline dualQuat operator / (dualQuat q){
        Quaternion denom = (q.real * q.real);
        Quaternion _real = (real * q.real)/denom;
        Quaternion _dual = ((q.real * dual)-(real * q.dual))/denom;
        return dualQuat(_real, _dual);
    }
    __host__ __device__
    inline void operator /= (dualQuat q){
        Quaternion denom = (q.real * q.real);
        Quaternion _real = (real * q.real)/denom;
        Quaternion _dual = ((q.real * dual)-(real * q.dual))/denom;
        *this = dualQuat(_real, _dual);
    }
    __host__ __device__
    inline dualQuat operator * (float scale){
        return dualQuat(real * scale, dual * scale);
    }
    inline void operator *= (float scale){
        *this = dualQuat(real * scale, dual * scale);
    }
    __host__ __device__ 
    inline dualQuat operator * (dualQuat q){
        return dualQuat(real * q.real, (real* q.dual) + (q.real * dual));
    }
    __host__ __device__ 
    inline void operator *= (dualQuat q){
        dualQuat dq(real * q.real, (real* q.dual) + (q.real * dual));
        *this = dq;
    }
    __host__ __device__ 
    inline  dualQuat operator + (dualQuat q){
        return dualQuat(real + q.real, dual + q.dual);
    }
    __host__ __device__ 
    inline  void operator += (dualQuat q){
        *this = dualQuat(real + q.real, dual + q.dual);
    }
    __host__ __device__ 
    inline  void operator = (dualQuat q){
        real = q.real; 
        dual = q.dual;
    }
    __host__ __device__ 
    inline  bool operator == (dualQuat q){
        bool res = (real == q.real && dual == q.dual) ? true : false;
        return res;
    }
    __host__ __device__ 
    inline  bool operator != (dualQuat q){
        bool res = (real != q.real || dual != q.dual) ? true : false;
        return res;
    }
    friend std::ostream& operator<<(std::ostream& os, dualQuat& q){
        os <<  "Real: " << q.real << std::endl;
        os <<  "Dual: " << q.dual ;
        return os;
    }
    /****************************************************************/
    /**************************Applications**************************/
    /****************************************************************/

    inline float3 rotate(float3 v){
        Quaternion q = real;
        q.normalize();
        return q.rotate(v);
    }
    __host__ __device__ 
    inline static Quaternion getRotation(dualQuat q){
        return q.real;
    }
    __host__ __device__ 
    inline Quaternion getRotation(void){
        return real;
    }
    __host__ __device__
    inline static float3 getTranslation(dualQuat q){
        Quaternion translation =  (q.dual * q.real.conjugate()) * 2.0f;
        return translation.getVectorPart();
    }
    __host__ __device__
    inline float3 getTranslation(void){
        Quaternion translation = (dual * real.conjugate()) * 2.0f;
        return translation.getVectorPart();
    }
    __host__ __device__
    inline static dualQuat addTranslation(dualQuat q, float3 t){
        return dualQuat(q.dual, t + q.getTranslation());
    }
    __host__ __device__
    inline void addTranslation(float3 t){
        dualQuat dq(real, t + this->getTranslation());
        *this = dq;
    }
    __host__ __device__ 
    inline static float4x4 getTransformation( dualQuat q ){
        q.normalize();
        float3x4 M;
        float w = q.real.w;
        float x = q.real.x;
        float y = q.real.y;
        float z = q.real.z; 
        // Extract rotational information
        M.m11 = powf(w, 2) + powf(x, 2) - powf(y, 2) - powf(z, 2);
        M.m12 = 2 * x * y + 2 * w * z;
        M.m13 = 2 * x * z - 2 * w * y;

        M.m21 = 2 * x * y - 2 * w * z;
        M.m22 = powf(w, 2) + powf(y, 2) - powf(x, 2) - powf(z, 2);
        M.m23 = 2 * y * z + 2 * w * x;
        M.m31 = 2 * x * z + 2 * w * y;
        M.m32 = 2 * y * z - 2 * w * x;
        M.m33 = powf(w, 2) + powf(z, 2) - powf(x, 2) - powf(y, 2);
        // Extract translation information
        Quaternion t = (q.dual * 2.0f) * Quaternion::conjugate( q.real);
        M.setTranslation(t.getVectorPart());
        return M;
    }
    __host__ __device__ 
    inline static float3 transformPosition(float3 point, dualQuat q ){
        float norm = q.real.norm();
        Quaternion blendReal = q.real / norm;
        Quaternion blendDual = q.dual / norm;

        float3 vecReal = Quaternion::getVectorPart(blendReal);
        float3 vecDual = Quaternion::getVectorPart(blendDual);
        float3 tranlation = ((vecDual * blendReal.w - vecReal * blendDual.w) + cross(vecReal, vecDual)) * 2.0f;
        return blendReal.rotate(point) + tranlation;

    }
    __host__ __device__ 
    inline float3 transformPosition(float3 point){
        float norm = real.norm();
        Quaternion blendReal(real / norm);
        Quaternion blendDual(dual / norm);
        float3 vecReal = Quaternion::getVectorPart(blendReal);
        float3 vecDual = Quaternion::getVectorPart(blendDual);
        float3 tranlation = ((vecDual * blendReal.w - vecReal * blendDual.w) + cross(vecReal, vecDual)) * 2.0f;
        return blendReal.rotate(point) + tranlation;

    }
    __host__ __device__ 
    inline static float3 transformNormal( float3 normal, dualQuat q ){
        float norm = q.real.norm();
        Quaternion blendReal = q.real / norm;
        Quaternion blendDual = q.dual / norm;

        float3 vecReal = Quaternion::getVectorPart(blendReal);
        float3 vecDual = Quaternion::getVectorPart(blendDual);
        float3 tranlation = ((vecDual * blendReal.w - vecReal * blendDual.w) + cross(vecReal, vecDual)) * 2.0f;
        return (blendReal.rotate(normal)) + tranlation;
    }
    __host__ __device__ 
    inline float3 transformNormal(float3 normal){
        float norm = real.norm();
        Quaternion blendReal = real / norm;
        Quaternion blendDual = dual / norm;

        float3 vecReal = Quaternion::getVectorPart(blendReal);
        float3 vecDual = Quaternion::getVectorPart(blendDual);
        float3 tranlation = ((vecDual * blendReal.w - vecReal * blendDual.w) + cross(vecReal, vecDual)) * 2.0f;
        return (blendReal.rotate(normal)) + tranlation;
    }
};

}   // namespace math
}   // namespace DynaMap