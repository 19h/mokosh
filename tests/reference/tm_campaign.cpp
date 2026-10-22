// Deterministic sequence campaign for the pinned htm.core reference.
#include <htm/algorithms/TemporalMemory.hpp>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
using namespace htm;
int main() {
  for(int seed=1;seed<=12;++seed) {
    TemporalMemory tm({16},4,3,0.4f,0.5f,2,3,0.1f,0.05f,0.02f,seed,4,8);
    unsigned state = 0x12345678u + seed;
    for(unsigned step=0;step<300;++step) {
      if(step%37==0) tm.reset();
      std::vector<UInt> columns;
      for(unsigned i=0;i<6;++i) { state=1664525u*state+1013904223u; columns.push_back((state>>16)%16); }
      std::sort(columns.begin(),columns.end());columns.erase(std::unique(columns.begin(),columns.end()),columns.end());
      SDR input({16});input.setSparse(columns);tm.compute(input,step%3!=0);tm.activateDendrites(false);
      std::cout << seed << ' ' << step;
      for(const auto &list:{tm.getActiveCells(),tm.getWinnerCells(),tm.getPredictiveCells().getSparse()}) {
        std::cout << " -1";for(auto c:list) std::cout << ' ' << c;
      }
      uint32_t hash=2166136261u;std::vector<uint32_t> fullGraph;
      auto mix=[&](uint32_t value){fullGraph.push_back(value);hash=(hash^value)*16777619u;};
      for(CellIdx cell=0;cell<64;++cell) {
        auto segments=tm.connections.segmentsForCell(cell); std::sort(segments.begin(),segments.end());
        mix(cell);mix(segments.size());
        for(auto segment:segments) {
          mix(segment);
          std::vector<std::pair<uint32_t,uint32_t>> synapses;
          for(auto syn:tm.connections.synapsesForSegment(segment)) {
            const auto& data=tm.connections.dataForSynapse(syn);uint32_t bits;
            auto permanence=data.permanence; std::memcpy(&bits,&permanence,4);
            synapses.emplace_back(data.presynapticCell,bits);
          }
          std::sort(synapses.begin(),synapses.end());mix(synapses.size());
          for(auto syn:synapses) {mix(syn.first);mix(syn.second);}
        }
      }
      std::cout << " -2 " << hash;
      if(std::getenv("MOKOSH_FULL_GRAPH")) { std::cout << " -3"; for(auto word:fullGraph) std::cout << ' ' << word; }
      std::cout << '\n';
    }
  }
}
