#include <htm/encoders/ScalarEncoder.hpp>
#include <htm/encoders/RandomDistributedScalarEncoder.hpp>
#include <htm/utils/SdrMetrics.hpp>
#include <iostream>
#include <iomanip>
using namespace htm;
template<class T> void emit(const std::string& key,const T& values) {
    std::cout<<key;for(auto x:values)std::cout<<' '<<x;std::cout<<'\n';
}
int main() {
    std::cout<<std::setprecision(17);
    ScalarEncoderParameters p;p.minimum=1e9;p.maximum=1e9+1;p.size=1024;p.activeBits=21;
    ScalarEncoder e(p);SDR out({1024});
    for(unsigned i=0;i<5;++i) {e.encode(1e9+i*0.001,out);emit("scalar64."+std::to_string(i),out.getSparse());}
    RDSE_Parameters rp;rp.size=1000;rp.activeBits=20;rp.resolution=.001f;rp.seed=42;
    RandomDistributedScalarEncoder r(rp);SDR rout({1000});
    for(unsigned i=0;i<5;++i) {r.encode(1e6+i*0.001,rout);emit("rdse64."+std::to_string(i),rout.getSparse());}
    Metrics m(std::vector<UInt>{8},3);SDR data({8});
    for(unsigned i=0;i<10;++i) {
        std::vector<UInt> bits;for(unsigned b=0;b<8;++b)if((b+i)%3==0)bits.push_back(b);
        data.setSparse(bits);m.addData(data);
        emit("metrics.sparsity."+std::to_string(i),std::vector<Real>{m.sparsity.sparsity,m.sparsity.min(),m.sparsity.max(),m.sparsity.mean(),m.sparsity.std()});
        emit("metrics.frequency."+std::to_string(i),m.activationFrequency.activationFrequency);
        emit("metrics.entropy."+std::to_string(i),std::vector<Real>{m.activationFrequency.entropy()});
        if(i>0)emit("metrics.overlap."+std::to_string(i),std::vector<Real>{m.overlap.overlap,m.overlap.min(),m.overlap.max(),m.overlap.mean(),m.overlap.std()});
    }
}
