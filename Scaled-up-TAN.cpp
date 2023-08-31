#include "STAN.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>
#include <string.h>
#include <algorithm>



STAN::STAN() :
trainingIsFinished_(false)
{
}

STAN::STAN(char* const *&, char* const *) :xxyDist_(), trainingIsFinished_(false),xxxyDist_()
{
    name_ = "STAN";
}

STAN::~STAN(void)
{
}

void STAN::reset(InstanceStream &is)
{
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    trainingIsFinished_ = false;

    parents_.resize(noCatAtts_);
    weight.resize(noCatAtts);
    for(int k=0; k<noCatAtts_; k++)
    {
        parents_[k].resize(2);
    }
    for(int k=0; k<noCatAtts_; k++)
    {
        for(int i=0; i<2; i++)
        {
            parents_[k][i].resize(noCatAtts_);
        }
    }
    for(int i=0;i<noCatAtts_;i++)
    {
        for(int j=0;j<2;j++)
        {
            for(int k=0;k<noCatAtts_;k++)
            {
                parents_[i][j][k]=NOPARENT;
            }
        }
    }
    bestCatAtt=-1;
    for(int i=0; i<noCatAtts; i++)
    {
        weight[i]=0;
    }
    xxyDist_.reset(is);
    xxxyDist_.reset(is);
}

void STAN::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); 
}

void STAN::initialisePass()
{
    assert(trainingIsFinished_ == false);
}

void STAN::train(const instance &inst)
{
    xxyDist_.update(inst);
    xxxyDist_.update(inst);
}

double STAN::H_one_condition(CategoricalAttribute x1)
{
    double m=0.0;
    for (CatValue v1 = 0; v1 < xxyDist_.getNoValues(x1); v1++)
    {
        for (CatValue y = 0; y < xxyDist_.getNoClasses(); y++)
        {
            m+=(-1)*xxyDist_.xyCounts.jointP(x1,v1,y)*log2(xxyDist_.xyCounts.p(x1,v1,y));
        }
    }
    return m;
}
double STAN::H_two_condition(CategoricalAttribute x1,CategoricalAttribute x2)
{
    double m=0.0;
    for(CatValue v1 = 0; v1 < xxyDist_.getNoValues(x1); v1++)
    {
        for(CatValue v2 = 0;v2 < xxyDist_.getNoValues(x2);v2++)
        {
            for (CatValue y = 0; y < xxyDist_.getNoClasses(); y++)
            {
                m+=(-1)*xxyDist_.jointP(x1,v1,x2,v2,y)*log2(xxyDist_.p(x1,v1,x2,v2,y));
            }
        }
    }
    return m;
}
double STAN::H_three_condition(CategoricalAttribute x1,CategoricalAttribute x2,CategoricalAttribute x3)
{
    double m=0.0;
    for(CatValue v1 = 0; v1 < xxyDist_.getNoValues(x1); v1++)
    {
        for(CatValue v2 = 0;v2 < xxyDist_.getNoValues(x2);v2++)
        {
            for(CatValue v3 = 0;v3 < xxyDist_.getNoValues(x3);v3++)
            {
                for(CatValue y = 0; y < xxyDist_.getNoClasses(); y++)
                {
                    m+=(-1)*xxxyDist_.jointP(x1,v1,x2,v2,x3,v3,y)*log2(xxxyDist_.p(x1,v1,x2,v2,x3,v3,y));
                }
            }
        }
    }
    return m;
}
double STAN::LL1(CategoricalAttribute x1,CategoricalAttribute root)
{
    double m=0.0;
    double m1=0.0;
    for(CatValue y = 0; y < xxyDist_.getNoClasses(); y++)
    {
        for(CatValue v1 = 0;v1 < xxyDist_.getNoValues(x1);v1++)
        {
            m1+=xxyDist_.xyCounts.jointP(x1,v1,y)*log2(xxyDist_.xyCounts.p(x1,v1,y));
        }
    }
    return m1;

}
double STAN::LL2(CategoricalAttribute x1,CategoricalAttribute root,CategoricalAttribute x2)
{
    double m=0.0;
    double m1=0.0;
    for(CatValue y = 0; y < xxyDist_.getNoClasses(); y++)
    {
        for(CatValue v1 = 0; v1 < xxyDist_.getNoValues(x1); v1++)
        {
            for(CatValue v2 = 0; v2 < xxyDist_.getNoValues(x2); v2++)
            {
                m1+=xxyDist_.jointP(x1,v1,x2,v2,y)*log2(xxyDist_.p(x1,v1,x2,v2,y));
            }
        }
        return m1;
    }
}

double STAN::LL3(CategoricalAttribute x1,CategoricalAttribute root,CategoricalAttribute x2,CategoricalAttribute x3)
{
    double m=0.0;
    double m1=0.0;
    for(CatValue y = 0; y < xxyDist_.getNoClasses(); y++)
    {
        for(CatValue v1 = 0; v1 < xxyDist_.getNoValues(x1); v1++)
        {
            for(CatValue v2 = 0; v2 < xxyDist_.getNoValues(x2); v2++)
            {
                for(CatValue v3 = 0; v3 < xxyDist_.getNoValues(x3); v3++)
                {
                    m1+=xxxyDist_.jointP(x1,v1,x2,v2,x3,v3,y)*log2(xxxyDist_.p(x1,v1,x2,v2,x3,v3,y));
                }
            }
        }
        return m1;

    }
}

int STAN::iscircle(int Graph[200][200])
{
    int inmap[noCatAtts_];
    int circle[noCatAtts_];
    int visited[noCatAtts_];
    int markcircle=0;
    int stacks[noCatAtts_];
    int top=0;
    for(int p=0; p<noCatAtts_; p++)
    {
        inmap[p]=-1;
        visited[p]=0;
        circle[p]=0;
    }
    for(int p=0; p<noCatAtts_; p++)
    {
        for(int q=0; q<noCatAtts_; q++)
        {
            if(Graph[p][q]==1)
            {
                inmap[p]=p;
                inmap[q]=q;
            }
        }
    }
    for(int p=0; p<noCatAtts_; p++)
    {
        for(int x1=0; x1<noCatAtts_; x1++)
        {
            stacks[x1]=0;
        }
        if(inmap[p]!=-1&&visited[p]==0)
        {
            stacks[++top]=inmap[p];
            visited[p]=1;
            while(top!=0)
            {
                int markneigh=0;
                int data, i;
                data = stacks[top];
                circle[data]=1;
                for(i = 0; i < noCatAtts_; i++)
                {
                    if(Graph[data][i] == 1 && visited[i] == 0)
                    {
                        visited[i] = 1;
                        stacks[++top]=i;
                        circle[i]=1;
                        break;

                    }
                    if(Graph[data][i] == 1 && visited[i] == 1&&circle[i]==1)
                    {
                        markcircle=1;
                        break;
                    }
                }
                if(i==noCatAtts_)
                {
                    top--;
                    circle[data]=0;
                }
                if(markcircle==1)
                    break;
            }
            if(markcircle==1)
                break;
        }
    }
    return markcircle;
}
bool cmp(cmiorder &c1,cmiorder &c2){
    return (c1.value) > (c2.value);
}

void STAN::classify(const instance &inst, std::vector<double> &classDist)
{
    std::vector<std::vector<double> >temp;
    temp.resize(noCatAtts_);
    for(int i=0; i<noCatAtts_; i++)
    {
        temp[i].resize(noClasses_);
    }

    for(int x=0;x<noCatAtts_;x++)
    {
        for (CatValue y = 0; y < noClasses_; y++)
        {
            temp[x][y] = xxxyDist_.xxyCounts.xyCounts.p(y)* (std::numeric_limits<double>::max() / 4.0);
        }
        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
        {
            const CategoricalAttribute parent1 = parents_[x][0][x1];
            const CategoricalAttribute parent2 = parents_[x][1][x1];
            for (CatValue y = 0; y < noClasses_; y++)
            {
                if (parent1==NOPARENT&&parent2==NOPARENT)
                {
                    temp[x][y] *= xxxyDist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
                }
                else if (parent1!=NOPARENT&&parent2==NOPARENT)
                {
                    const InstanceCount totalCount1 = xxxyDist_.xxyCounts.xyCounts.getCount(parent1, inst.getCatVal(parent1));
                    if (totalCount1 == 0)
                    {

                        temp[x][y] *= xxxyDist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    }
                    else
                    {
                        temp[x][y] *= xxxyDist_.xxyCounts.p(x1, inst.getCatVal(x1), parent1, inst.getCatVal(parent1), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                    }
                }
                else if (parent1!=NOPARENT&&parent2!=NOPARENT)
                {
                    const InstanceCount totalCount1 = xxxyDist_.xxyCounts.getCount(parent1, inst.getCatVal(parent1), parent2, inst.getCatVal(parent2));
                    if (totalCount1 == 0)
                    {
                        const InstanceCount totalCount2 = xxxyDist_.xxyCounts.xyCounts.getCount(parent1, inst.getCatVal(parent1));
                        const InstanceCount totalCount3 = xxxyDist_.xxyCounts.xyCounts.getCount(parent2, inst.getCatVal(parent2));
                        if (totalCount3 > 0&& totalCount2>0)
                        {
                            if(xxxyDist_.xxyCounts.p(x1, inst.getCatVal(x1),parent2,inst.getCatVal(parent2),y)>xxxyDist_.xxyCounts.p(x1, inst.getCatVal(x1), parent1, inst.getCatVal(parent1), y))
                                temp[x][y] *= xxxyDist_.xxyCounts.p(x1, inst.getCatVal(x1),parent2,inst.getCatVal(parent2),y);
                            else
                                temp[x][y] *= xxxyDist_.xxyCounts.p(x1, inst.getCatVal(x1), parent1, inst.getCatVal(parent1), y);
                        }
                        else  if(totalCount3 == 0&&totalCount2>0)
                        {
                            temp[x][y] *= xxxyDist_.xxyCounts.p(x1, inst.getCatVal(x1), parent1, inst.getCatVal(parent1), y);
                        }
                        else if(totalCount3>0&&totalCount2==0)
                        {
                            temp[x][y] *= xxxyDist_.xxyCounts.p(x1, inst.getCatVal(x1),parent2,inst.getCatVal(parent2),y);
                        }
                        else
                        {
                            temp[x][y] *= xxxyDist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                        }
                    }
                    else
                    {
                        temp[x][y] *= xxxyDist_.p(x1, inst.getCatVal(x1), parent1, inst.getCatVal(parent1),parent2, inst.getCatVal(parent2), y);
                    }
                }
            }
        }
    }
    for(CatValue y = 0; y < noClasses_; y++)
    {
        double m=0.0;
        for(unsigned int x1 = 0; x1 < noCatAtts_; x1++)
        {
            m+=weight[x1]*temp[x1][y];
        }
        classDist[y]=m;
    }

   normalise(classDist);

}

void STAN::finalisePass()
{
    assert(trainingIsFinished_ == false);
    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(xxyDist_, cmi);//O(km2v2)
    CategoricalAttribute topcandidate,firstparent,topcandidate1,firstparent1;


    CategoricalAttribute firstAtt ;

    std::set<CategoricalAttribute> available;
    std::set<CategoricalAttribute> inmap;

    double max1=-std::numeric_limits<float>::max();

    int Graph[200][200];
    double value[noCatAtts_];
    int counts[noCatAtts_];
    int length=noCatAtts_*noCatAtts_;
    struct cmiorder cmi_inorder[length];
    int i=0;
    for(int x1=1; x1<noCatAtts_; x1++)//O(m2)
    {
        for(int x2=0; x2<x1; x2++)
        {
            double dd=cmi[x1][x2];
            cmi_inorder[i].value=dd;
            cmi_inorder[i].to=x2;
            cmi_inorder[i].parent=x1;
            i++;
        }
    }
    std::sort(cmi_inorder,cmi_inorder+i,cmp);
   for(int x=0;x<noCatAtts_;x++)
   {
        inmap.clear();
        available.clear();
        firstAtt=x;
        parents_[x][0][firstAtt]=NOPARENT;
        inmap.insert(firstAtt);
        double thre=-std::numeric_limits<float>::max();
        for(unsigned int x1=0; x1<noCatAtts_; x1++)
        {
            if(x1!=firstAtt)
            {
                 available.insert(x1);
            }
        }
        for(int i=0; i<noCatAtts_; i++)
        {
            for(int j=0; j<noCatAtts_; j++)
            {
                Graph[i][j]=-1;
            }
        }
        for(int i=0; i<noCatAtts_; i++)
        {
            counts[i]=0;
            value[i]=-1;
        }
        while(!available.empty())
        {
            topcandidate=-1;
            firstparent=-1;
            max1=(-1)*(std::numeric_limits<double>::max() / 2.0);
            for(std::set<CategoricalAttribute>::iterator x1=inmap.begin(); x1!=inmap.end(); x1++)
            {
                for(std::set<CategoricalAttribute>::iterator x2=available.begin(); x2!=available.end(); x2++)
                {
                    if(cmi[*x1][*x2]>max1)
                    {
                        max1=cmi[*x1][*x2];
                        topcandidate=*x2;
                        firstparent=*x1;
                    }
                }
            }
            if(topcandidate!=-1&&firstparent!=-1)
            {
                inmap.insert(topcandidate);
                available.erase(topcandidate);
                parents_[x][0][topcandidate]=firstparent;
                Graph[firstparent][topcandidate]=1;
            }
            else
            {
                std::set<CategoricalAttribute>::iterator m=available.begin();
                inmap.insert(*m);
                available.erase(*m);
            }
        }

        for(unsigned int k = 0; k < i; k++)//O(m4)  O(km2v3)
        {
            topcandidate1=-1;
            int x1=cmi_inorder[k].parent;
            int topcandidate1=cmi_inorder[k].to;
            if(topcandidate1!=x1)
            {
                if(x1==parents_[x][0][topcandidate1]||topcandidate1==parents_[x][0][x1])
                {
                    continue;
                }
                double m1=0.0,m2=0.0;
                int markcircle;
                if((parents_[x][1][x1]==NOPARENT)&&(parents_[x][1][topcandidate1]==NOPARENT))
                {
                    if((parents_[x][0][x1]!=NOPARENT)&&(parents_[x][0][topcandidate1]!=NOPARENT))
                    {
                        m1=H_three_condition(x1,parents_[x][0][x1],topcandidate1)+H_two_condition(topcandidate1,parents_[x][0][topcandidate1]);
                    }
                    else if((parents_[x][0][x1]==NOPARENT)&&(parents_[x][0][topcandidate1]!=NOPARENT))
                    {
                        m1=H_two_condition(x1,topcandidate1)+H_two_condition(topcandidate1,parents_[x][0][topcandidate1]);
                    }
                    else if((parents_[x][0][x1]!=NOPARENT)&&(parents_[x][0][topcandidate1]==NOPARENT))
                    {
                        m1=H_three_condition(x1,parents_[x][0][x1],topcandidate1)+H_one_condition(topcandidate1);
                    }
                    else if((parents_[x][0][x1]==NOPARENT)&&(parents_[x][0][topcandidate1]==NOPARENT))
                    {
                        m1=H_two_condition(x1,topcandidate1)+H_one_condition(topcandidate1);
                    }


                    if((parents_[x][0][topcandidate1]!=NOPARENT)&&parents_[x][0][x1]!=NOPARENT)
                    {
                        m2=H_three_condition(topcandidate1,parents_[x][0][topcandidate1],x1)+H_two_condition(x1,parents_[x][0][x1]);
                    }
                    else if((parents_[x][0][topcandidate1]==NOPARENT)&&(parents_[x][0][x1]!=NOPARENT))
                    {
                        m2=H_two_condition(topcandidate1,x1)+H_two_condition(x1,parents_[x][0][x1]);
                    }
                    else if((parents_[x][0][topcandidate1]!=NOPARENT)&&(parents_[x][0][x1]==NOPARENT))
                    {
                        m2=H_three_condition(topcandidate1,parents_[x][0][topcandidate1],x1)+H_one_condition(x1);
                    }
                    else if((parents_[x][0][topcandidate1]==NOPARENT)&&(parents_[x][0][x1]==NOPARENT))
                    {
                        m2=H_two_condition(topcandidate1,x1)+H_one_condition(x1);
                    }

                    if(m1<m2)
                    {
                        Graph[topcandidate1][x1]=1;
                        markcircle=iscircle(Graph);
                        if(markcircle==0)
                        {
                            if(parents_[x][0][x1]==NOPARENT)
                                parents_[x][0][x1]=topcandidate1;
                            else
                                parents_[x][1][x1]=topcandidate1;
                        }
                        else
                        {
                            Graph[topcandidate1][x1]=2;
                        }
                    }
                    else
                    {
                        Graph[x1][topcandidate1]=1;
                        markcircle=iscircle(Graph);
                        if(markcircle==0)
                        {
                            if(parents_[x][0][topcandidate1]==NOPARENT)
                                parents_[x][0][topcandidate1]=x1;
                            else
                                parents_[x][1][topcandidate1]=x1;
                        }
                        else
                        {
                            Graph[x1][topcandidate1]=2;
                       }
                    }
                }
            }
        }
        double sum=0;

    }


    for(int x=0; x<noCatAtts_; x++)//O(km2V3)
    {
        double m1=0.0;
        for(CatValue y = 0; y < xxyDist_.getNoClasses(); y++)
        {
             m1+=xxyDist_.xyCounts.p(y)*log2(xxyDist_.xyCounts.p(y));
        }
        double m=0.0;
        for(int x1=0; x1<noCatAtts_; x1++)
        {
            if(parents_[x][1][x1]!=NOPARENT&&parents_[x][0][x1]!=NOPARENT)
            {
                m+=H_three_condition(x1,parents_[x][0][x1],parents_[x][1][x1]);
            }
            else if(parents_[x][0][x1]!=NOPARENT&&parents_[x][1][x1]==NOPARENT)
            {
                m+=H_two_condition(x1,parents_[x][0][x1]);
            }
            else
            {
                m+=H_one_condition(x1);
            }
        }
        weight[x]=-(m1+m);
    }
    normalise(weight);
    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool STAN::trainingIsFinished()
{
    return trainingIsFinished_;
}

