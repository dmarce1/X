#include "defs.hpp"
#include "hydro.hpp"
#include "state.hpp"


EXPORT_GLOBAL
void hydro_x_boundaries(real* U, int i1, int i2, int nx, int ny, int nz) {
	int j, k;
	j = i1;
	k = i2;

	const int D = nx * ny * ny;
	const int i0 = BW + nx * (j + ny * k);
	const int j0 = (nx - BW - 1) + nx * (j + ny * k);
	for (int f = 0; f != NF; ++f) {
		for (int i = 1; i <= BW; ++i) {
			U[i0 - i + D * f] = U[i0 + D * f];
			U[j0 + i + D * f] = U[j0 + D * f];
			U[momx_i * D + i0 - i] = min(U[momx_i * D + i0], zero);
			U[momx_i * D + j0 + i] = max(U[momx_i * D + j0], zero);
		}
	}
}

EXPORT_GLOBAL
void hydro_y_boundaries(real* U, int i1, int i2, int nx, int ny, int nz) {
	int i, k;
	i = i1;
	k = i2;
	const int D = nx * ny * ny;
	const int i0 = i + nx * (BW + ny * k);
	const int j0 = i + nx * ((ny - BW - 1) + ny * k);
	for (int f = 0; f != NF; ++f) {
		for (int j = 1; j <= BW; ++j) {
			U[i0 - j * nx + D * f] = U[i0 + D * f];
			U[j0 + j * nx + D * f] = U[j0 + D * f];
			U[i0 - j * ny + momy_i * D] = min(U[i0 + momy_i * D], zero);
			U[j0 + j * ny + momy_i * D] = max(U[j0 + momy_i * D], zero);
		}
	}
}

EXPORT_GLOBAL
void hydro_z_boundaries(real* U, int i1, int i2, int nx, int ny, int nz) {
	const int D = nx * ny * ny;
	int i, j;
	i = i1;
	j = i2;
	const int i0 = i + nx * (j + ny * BW);
	const int j0 = i + nx * (j + ny * (nz - BW - 1));
	for (int f = 0; f != NF; ++f) {
		for (int k = 1; k <= BW; ++k) {
			U[i0 - k * nx * ny + D * f] = U[i0 + D * f];
			U[j0 + k * nx * ny + D * f] = U[j0 + D * f];
			U[i0 - k * nx * ny + momz_i * D] = min(U[i0 + momz_i * D], zero);
			U[j0 + k * nx * ny + momz_i * D] = max(U[j0 + momz_i * D], zero);
		}
	}
}
