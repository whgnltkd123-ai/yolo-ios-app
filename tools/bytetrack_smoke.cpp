#include "OfficialByteTrack.h"
#include <iostream>
#include <stdexcept>
static void check(bool ok) { if (!ok) throw std::runtime_error("ByteTrack integration regression"); }
int main() {
  void *engine=obt_create();
  OBTTrack output[50]={};
  int id=-1;
  for(int f=0;f<40;++f) {
    OBTDetection d={100.0f+f,100,30,30,f>=8 && f<=12 ? 0.3f:0.9f,7};
    int n=obt_update(engine,(f==15 || f==16)?nullptr:&d,(f==15 || f==16)?0:1,output,50);
    if(f==15 || f==16) { check(n==0); continue; }
    check(n==1 && output[0].index==7);
    if(id<0) id=output[0].id;
    check(output[0].id==id);
  }
  obt_destroy(engine);
  engine=obt_create();
  OBTDetection d={100,100,30,30,0.3f,0};
  for(int i=0;i<10;++i) check(obt_update(engine,&d,1,output,50)==0);
  obt_destroy(engine);
  std::cout<<"NATIVE_BYTETRACK_SMOKE_PASS_NOT_DEVICE_PERFORMANCE_PROOF\n";
}
