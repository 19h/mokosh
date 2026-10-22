// Behavioral oracle for htm.core d86999487036c36c3b242ee46b9ba9b8f566087d.
#include <htm/algorithms/SpatialPooler.hpp>
#include <htm/algorithms/TemporalMemory.hpp>
#include <htm/algorithms/SDRClassifier.hpp>
#include <htm/encoders/ScalarEncoder.hpp>
#include <htm/encoders/DateEncoder.hpp>
#include <cstdlib>
#include <ctime>
#include <htm/encoders/RandomDistributedScalarEncoder.hpp>
#include <htm/encoders/SimHashDocumentEncoder.hpp>
#include <htm/utils/Random.hpp>
#include <htm/utils/Topology.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace htm;
template<class C> void row(const std::string &key, const C &values) {
  std::cout << key;
  for (const auto &v : values) std::cout << ' ' << v;
  std::cout << '\n';
}
SDR input(UInt size, std::initializer_list<UInt> bits) {
  SDR s({size}); std::vector<UInt> v(bits); s.setSparse(v); return s;
}
int main() {
  std::cout << std::setprecision(17);
  Random rng(42); std::vector<UInt> words;
  for(int i=0;i<1000;++i) words.push_back(rng.getUInt32());
  row("rng.words", words);
  Random floats(42); std::vector<double> reals;
  for(int i=0;i<5;++i) reals.push_back(floats.getReal64());
  row("rng.reals", reals);
  for(bool global: {true,false}) {
    SpatialPooler sp({40},{20},40,0.8,global,0.2,0,0,0.008,0.05,0.1,0.001,1000,0.0,1,0,true);
    auto in = input(40,{1,3,8,14,20,27,32}); SDR out({20});
    for(int step=0;step<5;++step) {
      const auto prefix = std::string(global?"sp.global.":"sp.local.")+std::to_string(step);
      row(prefix+".overlaps",sp.compute(in,true,out)); row(prefix+".active",out.getSparse());
      std::vector<Real> duty(20); sp.getActiveDutyCycles(duty.data()); row(prefix+".duty",duty);
    }
  }
  {
    SpatialPooler sp({8,4},{4,8},1,1.0,false,0.25,0,2,0.008,0.05,0.1,0.001,1000,2.0,1,0,true);
    SDR in({8,4}),out({4,8});
    for(int step=0;step<60;++step) {
      std::vector<UInt> bits;
      for(UInt b=0;b<32;++b) if((b+step)%3==0) bits.push_back(b);
      in.setSparse(bits);
      auto overlap=sp.compute(in,true,out);
      if(step==0 || step==1 || step==49 || step==50 || step==59) {
        auto key=std::string("sp.geometry.")+std::to_string(step);
        row(key+".overlaps",overlap);row(key+".active",out.getSparse());
        row(key+".radius",std::vector<UInt>{sp.getInhibitionRadius()});
        std::vector<Real> boost(32);sp.getBoostFactors(boost.data());row(key+".boost",boost);
      }
    }
  }
  row("reference_bug.nonwrapping_neighbors", Neighborhood::updateAllNeighbors(1, {4,8}, false, true).at(31));
  ScalarEncoderParameters p; p.minimum=0; p.maximum=10; p.size=13; p.activeBits=3;
  ScalarEncoder scalar(p); SDR encoded({13});
  for(double x: {0.,0.6,1.4,5.,9.6,10.}) {
    scalar.encode(x,encoded); row("scalar."+std::to_string(x),encoded.getSparse());
  }
  RDSE_Parameters rp; rp.size=1000;rp.activeBits=40;rp.resolution=1;rp.seed=42;
  RandomDistributedScalarEncoder rdse(rp); SDR re({1000});
  for(double x:{0.,1.,10.,20.}) { rdse.encode(x,re); row("rdse."+std::to_string(x),re.getSparse()); }
  SimHashDocumentEncoderParameters hp;hp.size=400;hp.activeBits=21;
  SimHashDocumentEncoder sh(hp);SDR he({400});
  sh.encode(std::vector<std::string>{"hello"},he);row("simhash.hello",he.getSparse());
  setenv("TZ", "UTC", 1); tzset();
  DateEncoderParameters dp; dp.season_width=3;dp.dayOfWeek_width=3;
  dp.weekend_width=3;dp.holiday_width=3;dp.timeOfDay_width=3;
  dp.custom_width=3;dp.custom_days={"mon", "fri"};
  DateEncoder date(dp); SDR de({date.size});
  for(std::time_t ts: {std::time_t(-1),std::time_t(0),std::time_t(1577836800),std::time_t(1608897600),std::time_t(1640973600)}) {
    date.encode(*std::gmtime(&ts),de); row("date."+std::to_string(ts),de.getSparse());
  }
  TemporalMemory tm({8},4,2,0.5,0.5,2,3,0.1,0.1,0.0,42,255,255,true,0,TemporalMemory::ANMode::RAW);
  auto a=input(8,{0,1,2}); auto b=input(8,{4,5,6});
  for(int epoch=0;epoch<8;++epoch) { tm.reset();tm.compute(a,true);tm.compute(b,true); }
  tm.reset();tm.compute(a,false);tm.activateDendrites(false);
  row("tm.predicted",tm.getPredictiveCells().getSparse());
  row("tm.winners",tm.getWinnerCells());
  tm.compute(b,false);row("tm.active",tm.getActiveCells());
  row("tm.anomaly",std::vector<Real>{tm.anomaly});
  Predictor predictor({0,1,3},0.1);
  for(UInt i=0;i<8;++i) {predictor.learn(2*i,a,i%2);}
  for(const auto &kv:predictor.infer(a)) row("classifier."+std::to_string(kv.first),kv.second);
}
