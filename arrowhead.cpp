#include "deps/json.hpp"
#include "hip/hip_runtime.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "utils.hpp"

using json = nlohmann::json;

std::random_device rd;
std::mt19937 gen(rd());

using VecOfVec = ContiguousVecOfVec<double>;

json
getConfig(std::string const& configFile)
{
  try {
    std::cout << "Reading config file\n";
    std::ifstream f(configFile);
    return json::parse(f);
  } catch (std::exception const& e) {
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
  int const deviceId = 0;
  gen.seed(1);

  hipSetDevice(deviceId);
  auto const props = getDeviceProps(deviceId);
  std::cout << props << std::endl;

  std::cout << std::setprecision(9);

  json resultObj;
  resultObj["results"] = json::array();

  int const trials = config["trials"];
  for (int const& nCells : config["cell_counts"]) {
    for (int const& nKcells : config["kcell_counts"]) {
      // xl value, this will store the first result
      std::vector<double> xlValues(nCells, 0);
      // xi value, this will store the second result
      auto xiValues = VecOfVec(nCells, nKcells);

      auto const allocateStart = std::chrono::high_resolution_clock::now();
      ArrowheadSystem<double> ahs(nCells, nKcells, gen);

      auto const solveStart = std::chrono::high_resolution_clock::now();
      for (int t = 0; t < trials; ++t) {
        ahs.solve(xlValues, xiValues);
      }
      auto const end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> const allocateTime =
        std::chrono::duration_cast<std::chrono::nanoseconds>(solveStart -
                                                             allocateStart);
      std::chrono::duration<double> const solveTime =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - solveStart);

      double const aveSolveTime = double(solveTime.count()) / double(trials);

      std::cout << "[" << nCells << ", " << nKcells
                << "] solve: " << aveSolveTime
                << " ns, allocate: " << allocateTime.count() << " ns\n";

      resultObj["results"].push_back(
        json({ { "n_kcells", nKcells },
               { "n_cells", nCells },
               { "solve_time", aveSolveTime },
               { "allocate_time", allocateTime.count() } }));

#ifdef DIAGNOSTIC
      auto const& xlExpected = ahs.xlExpected();
      auto const& xiExpected = ahs.xiExpected();
      for (int c = 0; c < nCells; ++c) {
        for (int k = 0; k < nKcells; ++k) {
          int const i = c * nKcells + k;
          std::cout << "[" << c << ", " << k << "]: Expected " << xiExpected[i]
                    << ", Actual: " << xiValues[i] << std::endl;
        }

        std::cout << "[" << c << ", " << nKcells << "]: Expected "
                  << xlExpected[c] << ", Actual: " << xlValues[c] << std::endl;
      }
#endif
    }
  }

  try {
    std::ofstream outFile;
    outFile.open(resultFile);
    outFile << resultObj.dump(2);
    outFile.close();
  } catch (std::exception const& e) {
    std::cout << "Could not write output file because exception:\n"
              << e.what() << std::endl;
    return 1;
  }
}