#include "hip/hip_runtime.h"
#include "deps/json.hpp"
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <fstream>

#include "utils.hpp"

using json = nlohmann::json;

std::random_device rd;
std::mt19937 gen(rd());

using VecOfVec = ContiguousVecOfVec<double>;

json getConfig(std::string const& configFile)
{
  try
  {
    std::cout << "Reading config file\n";
    std::ifstream f(configFile);
    return json::parse(f);
  }
  catch (std::exception const& e)
  {
    std::cout << "Could not read config file " << configFile
              << " because of exception:\n"
              << e.what() << std::endl;
    return 1;
  }
}

int
main()
{
  std::string const configFile = "config.json";
  std::string const resultFile = "results.json";

  auto const config = getConfig(configFile);

  int const nCells = config["cell_counts"][0];
  int const nKcells = config["kcell_counts"][0];
  int const deviceId = 0;
  gen.seed(1);

  hipSetDevice(deviceId);
  auto const props = getDeviceProps(deviceId);
  std::cout << props << std::endl;

  // xl value, this will store the first result
  std::vector<double> xlValues(nCells, 0);
  // xi value, this will store the second result
  auto xiValues = VecOfVec(nCells, nKcells);

  gen.seed(1);
  ArrowheadSystem<double> ahs(nCells, nKcells, gen);

  ahs.solve(xlValues, xiValues);

  auto const& xlExpected = ahs.xlExpected();
  auto const& xiExpected = ahs.xiExpected();


#ifdef DIAGNOSTIC
  for (int c = 0; c < nCells; ++c) {
    for (int k = 0; k < nKcells; ++k) {
      int const i = c * nKcells + k;
      std::cout << "[" << c << ", " << k << "]: Expected " << xiExpected[i]
                << ", Actual: " << xiValues[i] << std::endl;
    }

    std::cout << "[" << c << ", " << nKcells << "]: Expected " << xlExpected[c]
              << ", Actual: " << xlValues[c] << std::endl;
  }
#endif
}