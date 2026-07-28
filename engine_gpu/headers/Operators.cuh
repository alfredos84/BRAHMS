/*---------------------------------------------------------------------------*/
// * This file contains a set of overloaded operators to deal with complex numbers.
/*---------------------------------------------------------------------------*/


#ifndef _OPERATORSCUH
#define _OPERATORSCUH

#pragma once


/////////////////////////////////////     OPERATORS     ////////////////////////////////////////
__host__ __device__ inline complex_t operator+(const real_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a   + b.x;
	c.y =     + b.y;
	
	return c;
}


__host__ __device__ inline complex_t operator+(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x = a   + b.x;
	c.y =     + b.y;
	
	return c;
}


__host__ __device__ inline complex_t operator+(const complex_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	
	return c;
}


__host__ __device__ inline complex_t& operator+=(complex_t &a, const complex_t &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}


__host__ __device__ inline complex_t operator-(const complex_t &a) {
	
	complex_t c;    
	c.x = -a.x;
	c.y = -a.y;
	
	return c;
}

__host__ __device__ inline complex_t operator-(const real_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a   - b.x;
	c.y =     - b.y;
	
	return c;
}


__host__ __device__ inline complex_t operator-(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x =  b.x - a ;
	c.y =  b.y ;
	
	return c;
}


__host__ __device__ inline complex_t operator-(const complex_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a.x - b.x;
	c.y = a.y - b.y;
	
	return c;
}


__host__ __device__ inline complex_t operator*(const real_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a * b.x ;
	c.y = a * b.y ;
	
	return c;
}


__host__ __device__ inline complex_t operator*(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x = a * b.x ;
	c.y = a * b.y ;
	
	return c;
}


__host__ __device__ inline complex_t operator*(const complex_t &a, const complex_t &b) {
	
	complex_t c;    
	c.x = a.x * b.x - a.y * b.y ;
	c.y = a.x * b.y + a.y * b.x ;
	
	return c;
}


__host__ __device__ inline complex_t& operator*=(complex_t &a, const real_t &b) {
    a.x *= b;
    a.y *= b;
    return a;
}


__host__ __device__ inline complex_t operator/(const complex_t &b, const real_t &a) {
	
	complex_t c;    
	c.x = b.x / a ;
	c.y = b.y / a ;
	
	return c;
}


__host__ __device__ inline complex_t operator/(const real_t& a, const complex_t& b) {

	real_t denominator = b.x * b.x + b.y * b.y;
	complex_t c;
	c.x = (+a * b.x) / denominator;
	c.y = (-a * b.y) / denominator;

	return c;
}


__host__ __device__ inline complex_t operator/(const complex_t& a, const complex_t& b) {

	real_t denominator = b.x * b.x + b.y * b.y;
	complex_t c;
	c.x = (a.x * b.x + a.y * b.y) / denominator;
	c.y = (a.y * b.x - a.x * b.y) / denominator;

	return c;
}


__host__ __device__ inline complex_t& operator/=(complex_t &a, const real_t &b) {
    a.x /= b;
    a.y /= b;
    return a;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

//* Complex exponential e^(i*a) */
__host__ __device__ complex_t CpxExp (real_t a)
{
	complex_t b;
	b.x = cosf(a) ;	b.y = sinf(a) ;
	
	return b;
}


//* Complex exponential e^(a+i*b) */
__host__ __device__ complex_t CpxExp (complex_t a)
{
	complex_t b;
	b.x = expf(a.x)*cosf(a.y) ;	b.y = expf(a.x)*sinf(a.y) ;
	
	return b;
}


//* Complex conjugate */
__host__ __device__ complex_t CpxConj (complex_t a)
{
	complex_t b;
	b.x = +a.x ; b.y = -a.y ;
	
	return b;
}


//* Complex absolute value  */
__host__ __device__ real_t CpxAbs (complex_t a)
{
	real_t b;
	b = sqrtf(a.x*a.x + a.y*a.y);
	
	return b;
}


//* Complex square absolute value */
__host__ __device__ real_t CpxAbs2 (complex_t a)
{
	real_t b;
	b = a.x*a.x + a.y*a.y;
	
	return b;
}



__host__ __device__ complex_t CpxSqrt(complex_t z)
{
    real_t magnitude = sqrtf(z.x * z.x + z.y * z.y);
    real_t real = sqrtf(0.5f * (magnitude + z.x));
    real_t imag = sqrtf(0.5f * (magnitude - z.x));

    if (z.y < 0)
        imag = -imag;

    return make_cuFloatComplex(real, imag);
	
}


//////////////////////////////
struct MulRealVecRealScalar // Functor V*s: V real vector and s real scalar
{
	real_t s;
	MulRealVecRealScalar(real_t N) {s = N;};
	__host__ __device__
	real_t operator()(real_t x1)
	{
		return x1*s;
	}
};


struct MulCpxVecRealScalar // This functor performs the a*A, a real
{
	real_t a;
	MulCpxVecRealScalar(real_t A)  {a=A;};

	__host__ __device__
	complex_t operator()(complex_t V1)
	{
		complex_t result; 
		result.x = a*V1.x; result.y = a*V1.y; 
		return result;
	}
};


struct MulCpxVecCpxScalar // This functor scales by N, with N as a complex constant
{
	complex_t Norm;
	MulCpxVecCpxScalar(complex_t N) {Norm = N;};
	__host__ __device__
	complex_t operator()(complex_t V1)
	{
		return make_float2(V1.x*Norm.x - V1.y*Norm.y,
						   V1.x*Norm.y + V1.y*Norm.x);
		
	}
};


struct DivVecRealScalar // This functor scales by N, with N as a complex constant
{
	real_t Norm;
	DivVecRealScalar(real_t N) {Norm = N;};
	__host__ __device__
	complex_t operator()(complex_t V1)
	{
		return make_float2(V1.x/Norm, V1.y/Norm);		
	}
};


// //////////////////////////////
// // Returns A *= c -> A = c*A, with A real vector and c real constant
rVech_t operator*=(rVech_t &rhs, const real_t realscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), MulRealVecRealScalar(realscalar));
    return rhs;
}


rVecd_t operator*=(rVecd_t &rhs, const real_t realscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), MulRealVecRealScalar(realscalar));
    return rhs;
}


// Returns A *= c -> A = c*A, with A complex vector and c real constant
cVech_t operator*=(cVech_t &rhs, const real_t realscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), MulCpxVecRealScalar(realscalar));
    return rhs;
}


cVecd_t operator*=(cVecd_t &rhs, const real_t realscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), MulCpxVecRealScalar(realscalar));
    return rhs;
}


// Returns B = c*A, A complex vector and c complex constant
cVech_t operator*(cVech_t &rhs, const complex_t complexscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), MulCpxVecCpxScalar(complexscalar));
    return rhs;
}


cVecd_t  operator*(cVecd_t &rhs, const complex_t complexscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), MulCpxVecCpxScalar(complexscalar));
    return rhs;
}

cVech_t operator*(const complex_t complexscalar, cVech_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), MulCpxVecCpxScalar(complexscalar));
    return rhs;
}


cVech_t operator*(const complex_t complexscalar, cVecd_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), MulCpxVecCpxScalar(complexscalar));
    return rhs;
}



// Returns A /= c -> A = A/c, with A complex vector and c real constant
cVech_t operator/=(cVech_t &rhs, const real_t realscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), DivVecRealScalar(realscalar));
    return rhs;
}


cVecd_t operator/=(cVecd_t &rhs, const real_t realscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), DivVecRealScalar(realscalar));
    return rhs;
}
//////////////////////////////


#endif // -> #ifdef _OPERATORSCUH