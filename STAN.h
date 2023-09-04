#ifndef STAN_H
#define STAN_H
#pragma once
#include "incrementalLearner.h"
#include "xxyDist.h"
#include "xxxyDist.h"
#include "crosstab.h"
#include <limits>
#include <xxyDistEager.h>
struct cmiorder
{
    int parent;
    int to;
    double value;
};
class STAN: public IncrementalLearner {
public:
	STAN();
	STAN(char* const *& argv, char* const * end);
	~STAN(void);

	void reset(InstanceStream &is);   
	void initialisePass(); 
	void train(const instance &inst); 
	void finalisePass(); 
	bool trainingIsFinished(); 
	void getCapabilities(capabilities &c);
	double H_one_condition(CategoricalAttribute x1);
	double H_two_condition(CategoricalAttribute x1,CategoricalAttribute x2);
	double H_three_condition(CategoricalAttribute x1,CategoricalAttribute x2,CategoricalAttribute x3);
	int iscircle(int Graph[200][200]);
	double LL1(CategoricalAttribute x1,CategoricalAttribute root);
	double LL2(CategoricalAttribute x1,CategoricalAttribute root,CategoricalAttribute x2);
	double LL3(CategoricalAttribute x1,CategoricalAttribute root,CategoricalAttribute x2,CategoricalAttribute x3);

	
	virtual void classify(const instance &inst, std::vector<double> &classDist);

private:
	unsigned int noCatAtts_;          
	unsigned int noClasses_;     
    xxyDistEager xxyDistEager_;

	InstanceStream* instanceStream_;
	std::vector<std::vector<std::vector<CategoricalAttribute> > > parents_;
	xxyDist xxyDist_;
	xxxyDist xxxyDist_;
    std::vector<double> weight;

	bool trainingIsFinished_; 
	int bestCatAtt;

	const static CategoricalAttribute NOPARENT = 0xFFFFFFFFUL; 
};

#endif // STAN_H
