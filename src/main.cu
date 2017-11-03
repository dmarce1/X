
#include "grid.hpp"

int main(void) {
	int nx = 128;
	int ny = nx;
	int nz = nx;
	grid G(nx, ny, nz, 1.0, 1.0, 1.0);
	G.initialize(SOD);
	real t = 0.0;
	int i;
	for (i = 0; i < 10; ++i) {
		std::string fname = std::string("X.") + std::to_string(i) + std::string(".silo");
	//	G.output(fname);
		auto dt = G.step(GPU);
		printf("%i %e %e\n", i, t, dt);
		t += dt;
	}
	std::string fname = std::string("X.") + std::to_string(i) + std::string(".silo");
	G.output(fname);

	return 0;
}

