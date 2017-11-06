#include "grid.hpp"

int main(void) {
	int nx = 200;
	int ny = nx;
	int nz = nx;
	grid G(nx, ny, nz, 4, 1.0 / nx, 1.0 / nx, 1.0 / nx);
	G.initialize(SOD);
	real t = 0.0;
	int i;
	for (i = 0; i < 10; ++i) {
	//	std::string fname = std::string("/tmp/X.") + std::to_string(i) + std::string(".silo");
	//	G.output(fname);
		auto dt = G.step();
		printf("%i %e %e\n", i, t, dt);
		t += dt;
	}
	//std::string fname = std::string("/tmp/X.") + std::to_string(i) + std::string(".silo");
	//G.output(fname);

	return 0;
}

