#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>

#include <cstdio>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename scalar_t>
__global__ void kernel_sh(
    const scalar_t * __restrict__ inputs, 
    scalar_t * outputs, 
    uint32_t B, uint32_t D, uint32_t C,
    scalar_t * dy_dx
) {
	const uint32_t b = threadIdx.x + blockIdx.x * blockDim.x;
	if (b >= B) return;

	const uint32_t C2 = C * C;

	// locate
	inputs += b * D;
	outputs += b * C2;

	scalar_t x = inputs[0], y = inputs[1], z = inputs[2];

	scalar_t xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z, xyz=xy*z;
	scalar_t x4=x2*x2, y4=y2*y2, z4=z2*z2;
	scalar_t x6=x4*x2, y6=y4*y2, z6=z4*z2;

	auto write_sh = [&]() {
		outputs[0] = 0.28209479177387814f ;                          // 1/(2*sqrt(pi))
		if (C <= 1) { return; }
		outputs[1] = -0.48860251190291987f*y ;                               // -sqrt(3)*y/(2*sqrt(pi))
		outputs[2] = 0.48860251190291987f*z ;                                // sqrt(3)*z/(2*sqrt(pi))
		outputs[3] = -0.48860251190291987f*x ;                               // -sqrt(3)*x/(2*sqrt(pi))
		if (C <= 2) { return; }
		outputs[4] = 1.0925484305920792f*xy ;                                // sqrt(15)*xy/(2*sqrt(pi))
		outputs[5] = -1.0925484305920792f*yz ;                               // -sqrt(15)*yz/(2*sqrt(pi))
		outputs[6] = 0.94617469575755997f*z2 - 0.31539156525251999f ;                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
		outputs[7] = -1.0925484305920792f*xz ;                               // -sqrt(15)*xz/(2*sqrt(pi))
		outputs[8] = 0.54627421529603959f*x2 - 0.54627421529603959f*y2 ;                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
		if (C <= 3) { return; }
		outputs[9] = 0.59004358992664352f*y*(-3.0f*x2 + y2) ;                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
		outputs[10] = 2.8906114426405538f*xy*z ;                             // sqrt(105)*xy*z/(2*sqrt(pi))
		outputs[11] = 0.45704579946446572f*y*(1.0f - 5.0f*z2) ;                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
		outputs[12] = 0.3731763325901154f*z*(5.0f*z2 - 3.0f) ;                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
		outputs[13] = 0.45704579946446572f*x*(1.0f - 5.0f*z2) ;                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
		outputs[14] = 1.4453057213202769f*z*(x2 - y2) ;                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
		outputs[15] = 0.59004358992664352f*x*(-x2 + 3.0f*y2) ;                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
		if (C <= 4) { return; }
		outputs[16] = 2.5033429417967046f*xy*(x2 - y2) ;                             // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
		outputs[17] = 1.7701307697799304f*yz*(-3.0f*x2 + y2) ;                                // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
		outputs[18] = 0.94617469575756008f*xy*(7.0f*z2 - 1.0f) ;                               // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
		outputs[19] = 0.66904654355728921f*yz*(3.0f - 7.0f*z2) ;                               // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
		outputs[20] = -3.1735664074561294f*z2 + 3.7024941420321507f*z4 + 0.31735664074561293f ;                                // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
		outputs[21] = 0.66904654355728921f*xz*(3.0f - 7.0f*z2) ;                               // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
		outputs[22] = 0.47308734787878004f*(x2 - y2)*(7.0f*z2 - 1.0f) ;                                // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
		outputs[23] = 1.7701307697799304f*xz*(-x2 + 3.0f*y2) ;                                // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
		outputs[24] = -3.7550144126950569f*x2*y2 + 0.62583573544917614f*x4 + 0.62583573544917614f*y4 ;                         // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		if (C <= 5) { return; }
		outputs[25] = 0.65638205684017015f*y*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                            // 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		outputs[26] = 8.3026492595241645f*xy*z*(x2 - y2) ;                           // 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
		outputs[27] = -0.48923829943525038f*y*(3.0f*x2 - y2)*(9.0f*z2 - 1.0f) ;                         // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
		outputs[28] = 4.7935367849733241f*xy*z*(3.0f*z2 - 1.0f) ;                              // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
		outputs[29] = 0.45294665119569694f*y*(14.0f*z2 - 21.0f*z4 - 1.0f) ;                             // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		outputs[30] = 0.1169503224534236f*z*(-70.0f*z2 + 63.0f*z4 + 15.0f) ;                            // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
		outputs[31] = 0.45294665119569694f*x*(14.0f*z2 - 21.0f*z4 - 1.0f) ;                             // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		outputs[32] = 2.3967683924866621f*z*(x2 - y2)*(3.0f*z2 - 1.0f) ;                               // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
		outputs[33] = -0.48923829943525038f*x*(x2 - 3.0f*y2)*(9.0f*z2 - 1.0f) ;                         // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
		outputs[34] = 2.0756623148810411f*z*(-6.0f*x2*y2 + x4 + y4) ;                         // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		outputs[35] = 0.65638205684017015f*x*(10.0f*x2*y2 - x4 - 5.0f*y4) ;                            // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		if (C <= 6) { return; }
		outputs[36] = 1.3663682103838286f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4) ;                               // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		outputs[37] = 2.3666191622317521f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                            // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		outputs[38] = 2.0182596029148963f*xy*(x2 - y2)*(11.0f*z2 - 1.0f) ;                             // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
		outputs[39] = -0.92120525951492349f*yz*(3.0f*x2 - y2)*(11.0f*z2 - 3.0f) ;                               // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
		outputs[40] = 0.92120525951492349f*xy*(-18.0f*z2 + 33.0f*z4 + 1.0f) ;                           // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
		outputs[41] = 0.58262136251873131f*yz*(30.0f*z2 - 33.0f*z4 - 5.0f) ;                            // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		outputs[42] = 6.6747662381009842f*z2 - 20.024298714302954f*z4 + 14.684485723822165f*z6 - 0.31784601133814211f ;                         // sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
		outputs[43] = 0.58262136251873131f*xz*(30.0f*z2 - 33.0f*z4 - 5.0f) ;                            // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		outputs[44] = 0.46060262975746175f*(x2 - y2)*(11.0f*z2*(3.0f*z2 - 1.0f) - 7.0f*z2 + 1.0f) ;                               // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
		outputs[45] = -0.92120525951492349f*xz*(x2 - 3.0f*y2)*(11.0f*z2 - 3.0f) ;                               // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
		outputs[46] = 0.50456490072872406f*(11.0f*z2 - 1.0f)*(-6.0f*x2*y2 + x4 + y4) ;                          // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		outputs[47] = 2.3666191622317521f*xz*(10.0f*x2*y2 - x4 - 5.0f*y4) ;                            // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		outputs[48] = 10.247761577878714f*x2*y4 - 10.247761577878714f*x4*y2 + 0.6831841051919143f*x6 - 0.6831841051919143f*y6 ;                         // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		if (C <= 7) { return; }
		outputs[49] = 0.70716273252459627f*y*(-21.0f*x2*y4 + 35.0f*x4*y2 - 7.0f*x6 + y6) ;                              // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
		outputs[50] = 5.2919213236038001f*xy*z*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4) ;                             // 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		outputs[51] = -0.51891557872026028f*y*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + 5.0f*x4 + y4) ;                          // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
		outputs[52] = 4.1513246297620823f*xy*z*(x2 - y2)*(13.0f*z2 - 3.0f) ;                           // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
		outputs[53] = -0.15645893386229404f*y*(3.0f*x2 - y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f) ;                              // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		outputs[54] = 0.44253269244498261f*xy*z*(-110.0f*z2 + 143.0f*z4 + 15.0f) ;                              // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
		outputs[55] = 0.090331607582517306f*y*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f) ;                              // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		outputs[56] = 0.068284276912004949f*z*(315.0f*z2 - 693.0f*z4 + 429.0f*z6 - 35.0f) ;                              // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
		outputs[57] = 0.090331607582517306f*x*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f) ;                              // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		outputs[58] = 0.07375544874083044f*z*(x2 - y2)*(143.0f*z2*(3.0f*z2 - 1.0f) - 187.0f*z2 + 45.0f) ;                         // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
		outputs[59] = -0.15645893386229404f*x*(x2 - 3.0f*y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f) ;                              // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		outputs[60] = 1.0378311574405206f*z*(13.0f*z2 - 3.0f)*(-6.0f*x2*y2 + x4 + y4) ;                         // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		outputs[61] = -0.51891557872026028f*x*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + x4 + 5.0f*y4) ;                          // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
		outputs[62] = 2.6459606618019f*z*(15.0f*x2*y4 - 15.0f*x4*y2 + x6 - y6) ;                               // 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		outputs[63] = 0.70716273252459627f*x*(-35.0f*x2*y4 + 21.0f*x4*y2 - x6 + 7.0f*y6) ;                              // 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))
	};

	write_sh();

	if (dy_dx) {
		scalar_t *dx = dy_dx + b * D * C2;
		scalar_t *dy = dx + C2;
		scalar_t *dz = dy + C2;

		auto write_sh_dx = [&]() {
			dx[0] = 0.0f ;                             // 0
			if (C <= 1) { return; }
			dx[1] = 0.0f ;                             // 0
			dx[2] = 0.0f ;                             // 0
			dx[3] = -0.48860251190291992f ;                          // -sqrt(3)/(2*sqrt(pi))
			if (C <= 2) { return; }
			dx[4] = 1.0925484305920792f*y ;                          // sqrt(15)*y/(2*sqrt(pi))
			dx[5] = 0.0f ;                             // 0
			dx[6] = 0.0f ;                             // 0
			dx[7] = -1.0925484305920792f*z ;                         // -sqrt(15)*z/(2*sqrt(pi))
			dx[8] = 1.0925484305920792f*x ;                          // sqrt(15)*x/(2*sqrt(pi))
			if (C <= 3) { return; }
			dx[9] = -3.5402615395598609f*xy ;                                // -3*sqrt(70)*xy/(4*sqrt(pi))
			dx[10] = 2.8906114426405538f*yz ;                                // sqrt(105)*yz/(2*sqrt(pi))
			dx[11] = 0.0f ;                            // 0
			dx[12] = 0.0f ;                            // 0
			dx[13] = 0.45704579946446572f - 2.2852289973223288f*z2 ;                          // sqrt(42)*(1 - 5*z2)/(8*sqrt(pi))
			dx[14] = 2.8906114426405538f*xz ;                                // sqrt(105)*xz/(2*sqrt(pi))
			dx[15] = -1.7701307697799304f*x2 + 1.7701307697799304f*y2 ;                               // 3*sqrt(70)*(-x2 + y2)/(8*sqrt(pi))
			if (C <= 4) { return; }
			dx[16] = 2.5033429417967046f*y*(3.0f*x2 - y2) ;                           // 3*sqrt(35)*y*(3*x2 - y2)/(4*sqrt(pi))
			dx[17] = -10.620784618679583f*xy*z ;                             // -9*sqrt(70)*xy*z/(4*sqrt(pi))
			dx[18] = 0.94617469575756008f*y*(7.0f*z2 - 1.0f) ;                         // 3*sqrt(5)*y*(7*z2 - 1)/(4*sqrt(pi))
			dx[19] = 0.0f ;                            // 0
			dx[20] = 0.0f ;                            // 0
			dx[21] = 0.66904654355728921f*z*(3.0f - 7.0f*z2) ;                         // 3*sqrt(10)*z*(3 - 7*z2)/(8*sqrt(pi))
			dx[22] = 0.94617469575756008f*x*(7.0f*z2 - 1.0f) ;                         // 3*sqrt(5)*x*(7*z2 - 1)/(4*sqrt(pi))
			dx[23] = 5.3103923093397913f*z*(-x2 + y2) ;                              // 9*sqrt(70)*z*(-x2 + y2)/(8*sqrt(pi))
			dx[24] = 2.5033429417967046f*x*(x2 - 3.0f*y2) ;                           // 3*sqrt(35)*x*(x2 - 3*y2)/(4*sqrt(pi))
			if (C <= 5) { return; }
			dx[25] = 13.127641136803401f*xy*(-x2 + y2) ;                             // 15*sqrt(154)*xy*(-x2 + y2)/(8*sqrt(pi))
			dx[26] = 8.3026492595241645f*yz*(3.0f*x2 - y2) ;                          // 3*sqrt(385)*yz*(3*x2 - y2)/(4*sqrt(pi))
			dx[27] = 2.9354297966115022f*xy*(1.0f - 9.0f*z2) ;                         // 3*sqrt(770)*xy*(1 - 9*z2)/(16*sqrt(pi))
			dx[28] = 4.7935367849733241f*yz*(3.0f*z2 - 1.0f) ;                         // sqrt(1155)*yz*(3*z2 - 1)/(4*sqrt(pi))
			dx[29] = 0.0f ;                            // 0
			dx[30] = 0.0f ;                            // 0
			dx[31] = 6.3412531167397574f*z2 - 9.5118796751096362f*z4 - 0.45294665119569694f ;                          // sqrt(165)*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
			dx[32] = 4.7935367849733241f*xz*(3.0f*z2 - 1.0f) ;                         // sqrt(1155)*xz*(3*z2 - 1)/(4*sqrt(pi))
			dx[33] = -13.209434084751759f*x2*z2 + 1.4677148983057511f*x2 + 13.209434084751759f*y2*z2 - 1.4677148983057511f*y2 ;                         // 3*sqrt(770)*(-9*x2*z2 + x2 + 9*y2*z2 - y2)/(32*sqrt(pi))
			dx[34] = 8.3026492595241645f*xz*(x2 - 3.0f*y2) ;                          // 3*sqrt(385)*xz*(x2 - 3*y2)/(4*sqrt(pi))
			dx[35] = 19.6914617052051f*x2*y2 - 3.2819102842008503f*x4 - 3.2819102842008503f*y4 ;                               // 15*sqrt(154)*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
			if (C <= 6) { return; }
			dx[36] = 4.0991046311514854f*y*(-10.0f*x2*y2 + 5.0f*x4 + y4) ;                             // 3*sqrt(6006)*y*(-10*x2*y2 + 5*x4 + y4)/(32*sqrt(pi))
			dx[37] = 47.332383244635047f*xy*z*(-x2 + y2) ;                           // 15*sqrt(2002)*xy*z*(-x2 + y2)/(8*sqrt(pi))
			dx[38] = 2.0182596029148963f*y*(3.0f*x2 - y2)*(11.0f*z2 - 1.0f) ;                           // 3*sqrt(91)*y*(3*x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
			dx[39] = 5.5272315570895412f*xy*z*(3.0f - 11.0f*z2) ;                              // 3*sqrt(2730)*xy*z*(3 - 11*z2)/(16*sqrt(pi))
			dx[40] = 0.92120525951492349f*y*(-18.0f*z2 + 33.0f*z4 + 1.0f) ;                             // sqrt(2730)*y*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
			dx[41] = 0.0f ;                            // 0
			dx[42] = 0.0f ;                            // 0
			dx[43] = 0.58262136251873131f*z*(30.0f*z2 - 33.0f*z4 - 5.0f) ;                              // sqrt(273)*z*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
			dx[44] = 0.92120525951492349f*x*(-18.0f*z2 + 33.0f*z4 + 1.0f) ;                             // sqrt(2730)*x*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
			dx[45] = -2.7636157785447706f*z*(x2 - y2)*(11.0f*z2 - 3.0f) ;                              // -3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
			dx[46] = 2.0182596029148963f*x*(x2 - 3.0f*y2)*(11.0f*z2 - 1.0f) ;                           // 3*sqrt(91)*x*(x2 - 3*y2)*(11*z2 - 1)/(8*sqrt(pi))
			dx[47] = 11.833095811158762f*z*(6.0f*x2*y2 - x4 - y4) ;                           // 15*sqrt(2002)*z*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
			dx[48] = 4.0991046311514854f*x*(-10.0f*x2*y2 + x4 + 5.0f*y4) ;                             // 3*sqrt(6006)*x*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
			if (C <= 7) { return; }
			dx[49] = 9.9002782553443485f*xy*(10.0f*x2*y2 - 3.0f*x4 - 3.0f*y4) ;                         // 21*sqrt(715)*xy*(10*x2*y2 - 3*x4 - 3*y4)/(32*sqrt(pi))
			dx[50] = 15.875763970811402f*yz*(-10.0f*x2*y2 + 5.0f*x4 + y4) ;                            // 9*sqrt(10010)*yz*(-10*x2*y2 + 5*x4 + y4)/(32*sqrt(pi))
			dx[51] = -10.378311574405206f*xy*(x2 - y2)*(13.0f*z2 - 1.0f) ;                             // -15*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(16*sqrt(pi))
			dx[52] = 4.1513246297620823f*yz*(3.0f*x2 - y2)*(13.0f*z2 - 3.0f) ;                          // 3*sqrt(385)*yz*(3*x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
			dx[53] = 0.93875360317376422f*xy*(66.0f*z2 - 143.0f*z4 - 3.0f) ;                            // 9*sqrt(35)*xy*(66*z2 - 143*z4 - 3)/(32*sqrt(pi))
			dx[54] = 0.44253269244498261f*yz*(-110.0f*z2 + 143.0f*z4 + 15.0f) ;                         // 3*sqrt(70)*yz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
			dx[55] = 0.0f ;                            // 0
			dx[56] = 0.0f ;                            // 0
			dx[57] = -12.194767023639836f*z2 + 44.714145753346067f*z4 - 38.752259652899923f*z6 + 0.45165803791258652f ;                         // sqrt(105)*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
			dx[58] = 0.44253269244498261f*xz*(-110.0f*z2 + 143.0f*z4 + 15.0f) ;                         // 3*sqrt(70)*xz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
			dx[59] = 30.97886890473422f*x2*z2 - 67.120882626924143f*x2*z4 - 1.4081304047606462f*x2 - 30.97886890473422f*y2*z2 + 67.120882626924143f*y2*z4 + 1.4081304047606462f*y2 ;                              // 9*sqrt(35)*(66*x2*z2 - 143*x2*z4 - 3*x2 - 66*y2*z2 + 143*y2*z4 + 3*y2)/(64*sqrt(pi))
			dx[60] = 4.1513246297620823f*xz*(x2 - 3.0f*y2)*(13.0f*z2 - 3.0f) ;                          // 3*sqrt(385)*xz*(x2 - 3*y2)*(13*z2 - 3)/(8*sqrt(pi))
			dx[61] = -0.51891557872026028f*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + 4.0f*x2*(x2 - 5.0f*y2) + x4 + 5.0f*y4) ;                              // -3*sqrt(385)*(13*z2 - 1)*(-10*x2*y2 + 4*x2*(x2 - 5*y2) + x4 + 5*y4)/(64*sqrt(pi))
			dx[62] = 15.875763970811402f*xz*(-10.0f*x2*y2 + x4 + 5.0f*y4) ;                            // 9*sqrt(10010)*xz*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
			dx[63] = -74.252086915082614f*x2*y4 + 74.252086915082614f*x4*y2 - 4.9501391276721742f*x6 + 4.9501391276721742f*y6 ;                         // 21*sqrt(715)*(-15*x2*y4 + 15*x4*y2 - x6 + y6)/(64*sqrt(pi))
		};

		auto write_sh_dy = [&]() {
			dy[0] = 0.0f ;                             // 0
			if (C <= 1) { return; }
			dy[1] = -0.48860251190291992f ;                          // -sqrt(3)/(2*sqrt(pi))
			dy[2] = 0.0f ;                             // 0
			dy[3] = 0.0f ;                             // 0
			if (C <= 2) { return; }
			dy[4] = 1.0925484305920792f*x ;                          // sqrt(15)*x/(2*sqrt(pi))
			dy[5] = -1.0925484305920792f*z ;                         // -sqrt(15)*z/(2*sqrt(pi))
			dy[6] = 0.0f ;                             // 0
			dy[7] = 0.0f ;                             // 0
			dy[8] = -1.0925484305920792f*y ;                         // -sqrt(15)*y/(2*sqrt(pi))
			if (C <= 3) { return; }
			dy[9] = -1.7701307697799304f*x2 + 1.7701307697799304f*y2 ;                                // 3*sqrt(70)*(-x2 + y2)/(8*sqrt(pi))
			dy[10] = 2.8906114426405538f*xz ;                                // sqrt(105)*xz/(2*sqrt(pi))
			dy[11] = 0.45704579946446572f - 2.2852289973223288f*z2 ;                          // sqrt(42)*(1 - 5*z2)/(8*sqrt(pi))
			dy[12] = 0.0f ;                            // 0
			dy[13] = 0.0f ;                            // 0
			dy[14] = -2.8906114426405538f*yz ;                               // -sqrt(105)*yz/(2*sqrt(pi))
			dy[15] = 3.5402615395598609f*xy ;                                // 3*sqrt(70)*xy/(4*sqrt(pi))
			if (C <= 4) { return; }
			dy[16] = 2.5033429417967046f*x*(x2 - 3.0f*y2) ;                           // 3*sqrt(35)*x*(x2 - 3*y2)/(4*sqrt(pi))
			dy[17] = 5.3103923093397913f*z*(-x2 + y2) ;                              // 9*sqrt(70)*z*(-x2 + y2)/(8*sqrt(pi))
			dy[18] = 0.94617469575756008f*x*(7.0f*z2 - 1.0f) ;                         // 3*sqrt(5)*x*(7*z2 - 1)/(4*sqrt(pi))
			dy[19] = 0.66904654355728921f*z*(3.0f - 7.0f*z2) ;                         // 3*sqrt(10)*z*(3 - 7*z2)/(8*sqrt(pi))
			dy[20] = 0.0f ;                            // 0
			dy[21] = 0.0f ;                            // 0
			dy[22] = 0.94617469575756008f*y*(1.0f - 7.0f*z2) ;                         // 3*sqrt(5)*y*(1 - 7*z2)/(4*sqrt(pi))
			dy[23] = 10.620784618679583f*xy*z ;                              // 9*sqrt(70)*xy*z/(4*sqrt(pi))
			dy[24] = 2.5033429417967046f*y*(-3.0f*x2 + y2) ;                          // 3*sqrt(35)*y*(-3*x2 + y2)/(4*sqrt(pi))
			if (C <= 5) { return; }
			dy[25] = 19.6914617052051f*x2*y2 - 3.2819102842008503f*x4 - 3.2819102842008503f*y4 ;                               // 15*sqrt(154)*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
			dy[26] = 8.3026492595241645f*xz*(x2 - 3.0f*y2) ;                          // 3*sqrt(385)*xz*(x2 - 3*y2)/(4*sqrt(pi))
			dy[27] = -1.4677148983057511f*(x2 - y2)*(9.0f*z2 - 1.0f) ;                         // -3*sqrt(770)*(x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
			dy[28] = 4.7935367849733241f*xz*(3.0f*z2 - 1.0f) ;                         // sqrt(1155)*xz*(3*z2 - 1)/(4*sqrt(pi))
			dy[29] = 6.3412531167397574f*z2 - 9.5118796751096362f*z4 - 0.45294665119569694f ;                          // sqrt(165)*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
			dy[30] = 0.0f ;                            // 0
			dy[31] = 0.0f ;                            // 0
			dy[32] = 4.7935367849733241f*yz*(1.0f - 3.0f*z2) ;                         // sqrt(1155)*yz*(1 - 3*z2)/(4*sqrt(pi))
			dy[33] = 2.9354297966115022f*xy*(9.0f*z2 - 1.0f) ;                         // 3*sqrt(770)*xy*(9*z2 - 1)/(16*sqrt(pi))
			dy[34] = 8.3026492595241645f*yz*(-3.0f*x2 + y2) ;                         // 3*sqrt(385)*yz*(-3*x2 + y2)/(4*sqrt(pi))
			dy[35] = 13.127641136803401f*xy*(x2 - y2) ;                              // 15*sqrt(154)*xy*(x2 - y2)/(8*sqrt(pi))
			if (C <= 6) { return; }
			dy[36] = 4.0991046311514854f*x*(-10.0f*x2*y2 + x4 + 5.0f*y4) ;                             // 3*sqrt(6006)*x*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
			dy[37] = 11.833095811158762f*z*(6.0f*x2*y2 - x4 - y4) ;                           // 15*sqrt(2002)*z*(6*x2*y2 - x4 - y4)/(32*sqrt(pi))
			dy[38] = 2.0182596029148963f*x*(x2 - 3.0f*y2)*(11.0f*z2 - 1.0f) ;                           // 3*sqrt(91)*x*(x2 - 3*y2)*(11*z2 - 1)/(8*sqrt(pi))
			dy[39] = -2.7636157785447706f*z*(x2 - y2)*(11.0f*z2 - 3.0f) ;                              // -3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
			dy[40] = 0.92120525951492349f*x*(-18.0f*z2 + 33.0f*z4 + 1.0f) ;                             // sqrt(2730)*x*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
			dy[41] = 0.58262136251873131f*z*(30.0f*z2 - 33.0f*z4 - 5.0f) ;                              // sqrt(273)*z*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
			dy[42] = 0.0f ;                            // 0
			dy[43] = 0.0f ;                            // 0
			dy[44] = 0.92120525951492349f*y*(18.0f*z2 - 33.0f*z4 - 1.0f) ;                              // sqrt(2730)*y*(18*z2 - 33*z4 - 1)/(32*sqrt(pi))
			dy[45] = 5.5272315570895412f*xy*z*(11.0f*z2 - 3.0f) ;                              // 3*sqrt(2730)*xy*z*(11*z2 - 3)/(16*sqrt(pi))
			dy[46] = -2.0182596029148963f*y*(3.0f*x2 - y2)*(11.0f*z2 - 1.0f) ;                          // -3*sqrt(91)*y*(3*x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
			dy[47] = 47.332383244635047f*xy*z*(x2 - y2) ;                            // 15*sqrt(2002)*xy*z*(x2 - y2)/(8*sqrt(pi))
			dy[48] = 4.0991046311514854f*y*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                              // 3*sqrt(6006)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
			if (C <= 7) { return; }
			dy[49] = -74.252086915082614f*x2*y4 + 74.252086915082614f*x4*y2 - 4.9501391276721742f*x6 + 4.9501391276721742f*y6 ;                         // 21*sqrt(715)*(-15*x2*y4 + 15*x4*y2 - x6 + y6)/(64*sqrt(pi))
			dy[50] = 15.875763970811402f*xz*(-10.0f*x2*y2 + x4 + 5.0f*y4) ;                            // 9*sqrt(10010)*xz*(-10*x2*y2 + x4 + 5*y4)/(32*sqrt(pi))
			dy[51] = 0.51891557872026028f*(13.0f*z2 - 1.0f)*(10.0f*x2*y2 - 5.0f*x4 + 4.0f*y2*(5.0f*x2 - y2) - y4) ;                                // 3*sqrt(385)*(13*z2 - 1)*(10*x2*y2 - 5*x4 + 4*y2*(5*x2 - y2) - y4)/(64*sqrt(pi))
			dy[52] = 4.1513246297620823f*xz*(x2 - 3.0f*y2)*(13.0f*z2 - 3.0f) ;                          // 3*sqrt(385)*xz*(x2 - 3*y2)*(13*z2 - 3)/(8*sqrt(pi))
			dy[53] = -0.46937680158688211f*(x2 - y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f) ;                             // -9*sqrt(35)*(x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
			dy[54] = 0.44253269244498261f*xz*(-110.0f*z2 + 143.0f*z4 + 15.0f) ;                         // 3*sqrt(70)*xz*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
			dy[55] = -12.194767023639836f*z2 + 44.714145753346067f*z4 - 38.752259652899923f*z6 + 0.45165803791258652f ;                         // sqrt(105)*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
			dy[56] = 0.0f ;                            // 0
			dy[57] = 0.0f ;                            // 0
			dy[58] = 0.44253269244498261f*yz*(110.0f*z2 - 143.0f*z4 - 15.0f) ;                          // 3*sqrt(70)*yz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
			dy[59] = 0.93875360317376422f*xy*(-66.0f*z2 + 143.0f*z4 + 3.0f) ;                           // 9*sqrt(35)*xy*(-66*z2 + 143*z4 + 3)/(32*sqrt(pi))
			dy[60] = -4.1513246297620823f*yz*(3.0f*x2 - y2)*(13.0f*z2 - 3.0f) ;                         // -3*sqrt(385)*yz*(3*x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
			dy[61] = 10.378311574405206f*xy*(x2 - y2)*(13.0f*z2 - 1.0f) ;                              // 15*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(16*sqrt(pi))
			dy[62] = 15.875763970811402f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                             // 9*sqrt(10010)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
			dy[63] = 9.9002782553443485f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4) ;                                // 21*sqrt(715)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		};

		auto write_sh_dz = [&]() {
			dz[0] = 0.0f ;                             // 0
			if (C <= 1) { return; }
			dz[1] = 0.0f ;                             // 0
			dz[2] = 0.48860251190291992f ;                           // sqrt(3)/(2*sqrt(pi))
			dz[3] = 0.0f ;                             // 0
			if (C <= 2) { return; }
			dz[4] = 0.0f ;                             // 0
			dz[5] = -1.0925484305920792f*y ;                         // -sqrt(15)*y/(2*sqrt(pi))
			dz[6] = 1.8923493915151202f*z ;                          // 3*sqrt(5)*z/(2*sqrt(pi))
			dz[7] = -1.0925484305920792f*x ;                         // -sqrt(15)*x/(2*sqrt(pi))
			dz[8] = 0.0f ;                             // 0
			if (C <= 3) { return; }
			dz[9] = 0.0f ;                             // 0
			dz[10] = 2.8906114426405538f*xy ;                                // sqrt(105)*xy/(2*sqrt(pi))
			dz[11] = -4.5704579946446566f*yz ;                               // -5*sqrt(42)*yz/(4*sqrt(pi))
			dz[12] = 5.597644988851731f*z2 - 1.1195289977703462f ;                            // 3*sqrt(7)*(5*z2 - 1)/(4*sqrt(pi))
			dz[13] = -4.5704579946446566f*xz ;                               // -5*sqrt(42)*xz/(4*sqrt(pi))
			dz[14] = 1.4453057213202769f*x2 - 1.4453057213202769f*y2 ;                                // sqrt(105)*(x2 - y2)/(4*sqrt(pi))
			dz[15] = 0.0f ;                            // 0
			if (C <= 4) { return; }
			dz[16] = 0.0f ;                            // 0
			dz[17] = 1.7701307697799304f*y*(-3.0f*x2 + y2) ;                          // 3*sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
			dz[18] = 13.246445740605839f*xy*z ;                              // 21*sqrt(5)*xy*z/(2*sqrt(pi))
			dz[19] = 2.0071396306718676f*y*(1.0f - 7.0f*z2) ;                          // 9*sqrt(10)*y*(1 - 7*z2)/(8*sqrt(pi))
			dz[20] = 14.809976568128603f*pow(z, 3) - 6.3471328149122579f*z ;                          // (105*z**3 - 45*z)/(4*sqrt(pi))
			dz[21] = 2.0071396306718676f*x*(1.0f - 7.0f*z2) ;                          // 9*sqrt(10)*x*(1 - 7*z2)/(8*sqrt(pi))
			dz[22] = 6.6232228703029197f*z*(x2 - y2) ;                               // 21*sqrt(5)*z*(x2 - y2)/(4*sqrt(pi))
			dz[23] = 1.7701307697799304f*x*(-x2 + 3.0f*y2) ;                          // 3*sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
			dz[24] = 0.0f ;                            // 0
			if (C <= 5) { return; }
			dz[25] = 0.0f ;                            // 0
			dz[26] = 8.3026492595241645f*xy*(x2 - y2) ;                              // 3*sqrt(385)*xy*(x2 - y2)/(4*sqrt(pi))
			dz[27] = 8.8062893898345074f*yz*(-3.0f*x2 + y2) ;                         // 9*sqrt(770)*yz*(-3*x2 + y2)/(16*sqrt(pi))
			dz[28] = 4.7935367849733241f*xy*(9.0f*z2 - 1.0f) ;                         // sqrt(1155)*xy*(9*z2 - 1)/(4*sqrt(pi))
			dz[29] = 12.682506233479513f*yz*(1.0f - 3.0f*z2) ;                         // 7*sqrt(165)*yz*(1 - 3*z2)/(4*sqrt(pi))
			dz[30] = -24.559567715218954f*z2 + 36.839351572828434f*z4 + 1.754254836801354f ;                           // 15*sqrt(11)*(-14*z2 + 21*z4 + 1)/(16*sqrt(pi))
			dz[31] = 12.682506233479513f*xz*(1.0f - 3.0f*z2) ;                         // 7*sqrt(165)*xz*(1 - 3*z2)/(4*sqrt(pi))
			dz[32] = 2.3967683924866621f*(x2 - y2)*(9.0f*z2 - 1.0f) ;                          // sqrt(1155)*(x2 - y2)*(9*z2 - 1)/(8*sqrt(pi))
			dz[33] = 8.8062893898345074f*xz*(-x2 + 3.0f*y2) ;                         // 9*sqrt(770)*xz*(-x2 + 3*y2)/(16*sqrt(pi))
			dz[34] = -12.453973889286246f*x2*y2 + 2.0756623148810411f*x4 + 2.0756623148810411f*y4 ;                            // 3*sqrt(385)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
			dz[35] = 0.0f ;                            // 0
			if (C <= 6) { return; }
			dz[36] = 0.0f ;                            // 0
			dz[37] = 2.3666191622317521f*y*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                              // 3*sqrt(2002)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
			dz[38] = 44.401711264127719f*xy*z*(x2 - y2) ;                            // 33*sqrt(91)*xy*z*(x2 - y2)/(4*sqrt(pi))
			dz[39] = -2.7636157785447706f*y*(3.0f*x2 - y2)*(11.0f*z2 - 1.0f) ;                          // -3*sqrt(2730)*y*(3*x2 - y2)*(11*z2 - 1)/(32*sqrt(pi))
			dz[40] = 11.054463114179082f*xy*z*(11.0f*z2 - 3.0f) ;                              // 3*sqrt(2730)*xy*z*(11*z2 - 3)/(8*sqrt(pi))
			dz[41] = 2.9131068125936568f*y*(18.0f*z2 - 33.0f*z4 - 1.0f) ;                               // 5*sqrt(273)*y*(18*z2 - 33*z4 - 1)/(16*sqrt(pi))
			dz[42] = 2.6699064952403937f*z*(-30.0f*z2 + 33.0f*z4 + 5.0f) ;                              // 21*sqrt(13)*z*(-30*z2 + 33*z4 + 5)/(16*sqrt(pi))
			dz[43] = 2.9131068125936568f*x*(18.0f*z2 - 33.0f*z4 - 1.0f) ;                               // 5*sqrt(273)*x*(18*z2 - 33*z4 - 1)/(16*sqrt(pi))
			dz[44] = 5.5272315570895412f*z*(x2 - y2)*(11.0f*z2 - 3.0f) ;                               // 3*sqrt(2730)*z*(x2 - y2)*(11*z2 - 3)/(16*sqrt(pi))
			dz[45] = -2.7636157785447706f*x*(x2 - 3.0f*y2)*(11.0f*z2 - 1.0f) ;                          // -3*sqrt(2730)*x*(x2 - 3*y2)*(11*z2 - 1)/(32*sqrt(pi))
			dz[46] = 11.10042781603193f*z*(-6.0f*x2*y2 + x4 + y4) ;                           // 33*sqrt(91)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
			dz[47] = 2.3666191622317521f*x*(10.0f*x2*y2 - x4 - 5.0f*y4) ;                              // 3*sqrt(2002)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
			dz[48] = 0.0f ;                            // 0
			if (C <= 7) { return; }
			dz[49] = 0.0f ;                            // 0
			dz[50] = 5.2919213236038001f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4) ;                                // 3*sqrt(10010)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
			dz[51] = 13.491805046726766f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4) ;                             // 39*sqrt(385)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
			dz[52] = 12.453973889286248f*xy*(x2 - y2)*(13.0f*z2 - 1.0f) ;                              // 9*sqrt(385)*xy*(x2 - y2)*(13*z2 - 1)/(8*sqrt(pi))
			dz[53] = -6.8841930899409371f*yz*(3.0f*x2 - y2)*(13.0f*z2 - 3.0f) ;                         // -33*sqrt(35)*yz*(3*x2 - y2)*(13*z2 - 3)/(16*sqrt(pi))
			dz[54] = 2.2126634622249131f*xy*(-66.0f*z2 + 143.0f*z4 + 3.0f) ;                            // 15*sqrt(70)*xy*(-66*z2 + 143*z4 + 3)/(32*sqrt(pi))
			dz[55] = 1.6259689364853116f*yz*(110.0f*z2 - 143.0f*z4 - 15.0f) ;                           // 9*sqrt(105)*yz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
			dz[56] = 64.528641681844675f*z2 - 236.60501950009714f*z4 + 205.05768356675085f*z6 - 2.3899496919201733f ;                           // 7*sqrt(15)*(135*z2 - 495*z4 + 429*z6 - 5)/(32*sqrt(pi))
			dz[57] = 1.6259689364853116f*xz*(110.0f*z2 - 143.0f*z4 - 15.0f) ;                           // 9*sqrt(105)*xz*(110*z2 - 143*z4 - 15)/(32*sqrt(pi))
			dz[58] = 0.07375544874083044f*(x2 - y2)*(143.0f*z2*(3.0f*z2 - 1.0f) + 132.0f*z2*(13.0f*z2 - 5.0f) - 187.0f*z2 + 45.0f) ;                         // sqrt(70)*(x2 - y2)*(143*z2*(3*z2 - 1) + 132*z2*(13*z2 - 5) - 187*z2 + 45)/(64*sqrt(pi))
			dz[59] = -6.8841930899409371f*xz*(x2 - 3.0f*y2)*(13.0f*z2 - 3.0f) ;                         // -33*sqrt(35)*xz*(x2 - 3*y2)*(13*z2 - 3)/(16*sqrt(pi))
			dz[60] = 3.1134934723215619f*(13.0f*z2 - 1.0f)*(-6.0f*x2*y2 + x4 + y4) ;                            // 9*sqrt(385)*(13*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
			dz[61] = 13.491805046726766f*xz*(10.0f*x2*y2 - x4 - 5.0f*y4) ;                             // 39*sqrt(385)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
			dz[62] = 39.6894099270285f*x2*y4 - 39.6894099270285f*x4*y2 + 2.6459606618019f*x6 - 2.6459606618019f*y6 ;                            // 3*sqrt(10010)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
			dz[63] = 0.0f ;                            // 0
		};
		write_sh_dx();
		write_sh_dy();
		write_sh_dz();
	}
}


template <typename scalar_t>
__global__ void kernel_sh_backward(
    const scalar_t * __restrict__ grad,
	const scalar_t * __restrict__ inputs,
    uint32_t B, uint32_t D, uint32_t C,
    const scalar_t * __restrict__ dy_dx,
    scalar_t * grad_inputs
) {
	const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t b = t / D;
	if (b >= B) return;

	const uint32_t d = t - b * D;
	const uint32_t C2 = C * C;

	// locate
	grad += b * C2;
	dy_dx += b * D * C2 + d * C2;

	for (int ch = 0; ch < C2; ch++) {
		grad_inputs[t] += grad[ch] * dy_dx[ch];
		//printf("t=%d, b=%d, d=%d, ch=%d, grad=%f (+= %f * %f)\n", t, b, d, ch, grad_inputs[t], grad[ch], dy_dx[ch]);
	}

}

// inputs: [B, D], float, in [0, 1]
// outputs: [B, L * C], float
template <typename scalar_t>
void sh_encode_forward_cuda(const scalar_t *inputs, scalar_t *outputs, const uint32_t B, const uint32_t D, const uint32_t C, scalar_t *dy_dx) {
	static constexpr uint32_t N_THREADS = 256;
	kernel_sh<scalar_t><<<div_round_up(B, N_THREADS), N_THREADS>>>(inputs, outputs, B, D, C, dy_dx);
}


template <typename scalar_t>
void sh_encode_backward_cuda(const scalar_t *grad, const scalar_t *inputs, const uint32_t B, const uint32_t D, const uint32_t C, scalar_t *dy_dx, scalar_t *grad_inputs) {
	static constexpr uint32_t N_THREADS = 256;
	kernel_sh_backward<scalar_t><<<div_round_up(B * D, N_THREADS), N_THREADS>>>(grad, inputs, B, D, C, dy_dx, grad_inputs);
}


void sh_encode_forward(at::Tensor inputs, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, at::optional<at::Tensor> dy_dx) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(outputs);
    // CHECK_CUDA(dy_dx);
    
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(outputs);
    // CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(outputs);
    // CHECK_IS_FLOATING(dy_dx);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    inputs.scalar_type(), "sh_encode_forward_cuda", ([&] {
		sh_encode_forward_cuda<scalar_t>(inputs.data_ptr<scalar_t>(), outputs.data_ptr<scalar_t>(), B, D, C, dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr);
    }));	
}

void sh_encode_backward(at::Tensor grad, at::Tensor inputs, const uint32_t B, const uint32_t D, const uint32_t C, at::Tensor dy_dx, at::Tensor grad_inputs) {    
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(dy_dx);
    CHECK_CUDA(grad_inputs);
    
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(dy_dx);
    CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(dy_dx);
    CHECK_IS_FLOATING(grad_inputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "sh_encode_backward_cuda", ([&] {
    	sh_encode_backward_cuda<scalar_t>(grad.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), B, D, C, dy_dx.data_ptr<scalar_t>(), grad_inputs.data_ptr<scalar_t>());
    }));	
}