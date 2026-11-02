#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include "mseloss.cpp"

// plain loop, the reference answer (double accumulator so it's the "true" value)
float referencemseloss(const float* pred,const float* actual,int size){
	double sum=0.0;
	for(int i=0;i<size;i++) sum+=fabs((double)pred[i]-(double)actual[i]);
	return (float)(sum/size);
}

// plain float loop for the speed comparison
float scalarmseloss(const float* pred,const float* actual,int size){
	float sum=0.0f;
	for(int i=0;i<size;i++) sum+=fabsf(pred[i]-actual[i]);
	return sum/size;
}

// same loop with vectorization forbidden: one element per iteration, one add chain
// icx defines __clang__, so it takes the first branch; GCC 14+ understands novector
float novecmseloss(const float* pred,const float* actual,int size){
	float sum=0.0f;
#if defined(__clang__)
	#pragma clang loop vectorize(disable) interleave(disable)
#elif defined(__GNUC__)
	#pragma GCC novector
#endif
	for(int i=0;i<size;i++) sum+=fabsf(pred[i]-actual[i]);
	return sum/size;
}

int main(){
	// small hand-checkable case: |1-0|+|-2-1|+|3-3|+|0.5-(-0.5)| = 1+3+0+1 = 5, /4 = 1.25
	float p[4]={1.0f,-2.0f,3.0f,0.5f};
	float a[4]={0.0f,1.0f,3.0f,-0.5f};
	printf("hand case: got %g, expected 1.25\n",mseloss(p,a,4));

	// sizes around the 8-wide SIMD boundary hit the tail loop: <8, ==8, 8+tail, many blocks
	int sizes[]={1,7,8,9,15,16,17,1000,1003};
	int failed=0;
	for(int size:sizes){
		vector<float> pred(size),actual(size);
		for(int i=0;i<size;i++){
			pred[i]=(float)((i*7)%11)-5.0f;      // mix of positive and negative
			actual[i]=(float)((i*3)%13)*0.5f-3.0f;
		}
		float got=mseloss(pred.data(),actual.data(),size);
		float expected=referencemseloss(pred.data(),actual.data(),size);
		float err=fabsf(got-expected);
		bool ok=err<1e-4f*fmaxf(1.0f,fabsf(expected));
		if(!ok) failed++;
		printf("size=%5d  got=%-10g expected=%-10g %s\n",size,got,expected,ok?"OK":"FAIL");
	}
	printf("%s\n",failed?"SOME TESTS FAILED":"all OK");

	// speed: your SIMD version vs a plain scalar loop the compiler is free to auto-vectorize
	// sizes: 4K fits L1, 256K fits L2, 4M fits L3, 64M is main memory
	printf("\n%10s %14s %14s %14s %10s\n","size","MAEloss ns/el","autovec ns/el","novec ns/el","MAE GB/s");
	int speedsizes[]={4096,262144,4194304,67108864};
	for(int size:speedsizes){
		vector<float> pred(size),actual(size);
		for(int i=0;i<size;i++){
			pred[i]=(float)((i*7)%11)-5.0f;
			actual[i]=(float)((i*3)%13)*0.5f-3.0f;
		}
		int reps=max(3,(int)(200000000LL/size));   // ~200M elements total per measurement
		volatile float sink=0.0f;                  // stops the compiler deleting the calls

		double bestmae=1e30,bestscalar=1e30,bestnovec=1e30;
		for(int trial=0;trial<3;trial++){
			auto t0=chrono::high_resolution_clock::now();
			for(int r=0;r<reps;r++) sink=sink+mseloss(pred.data(),actual.data(),size);
			auto t1=chrono::high_resolution_clock::now();
			for(int r=0;r<reps;r++) sink=sink+scalarmseloss(pred.data(),actual.data(),size);
			auto t2=chrono::high_resolution_clock::now();
			for(int r=0;r<reps;r++) sink=sink+novecmseloss(pred.data(),actual.data(),size);
			auto t3=chrono::high_resolution_clock::now();
			bestmae=min(bestmae,chrono::duration<double>(t1-t0).count());
			bestscalar=min(bestscalar,chrono::duration<double>(t2-t1).count());
			bestnovec=min(bestnovec,chrono::duration<double>(t3-t2).count());
		}
		double elems=(double)size*reps;
		printf("%10d %14.3f %14.3f %14.3f %10.1f\n",size,bestmae/elems*1e9,bestscalar/elems*1e9,
		       bestnovec/elems*1e9,elems*2*sizeof(float)/bestmae/1e9);   // 2 floats read per element
	}
	return failed;
}
