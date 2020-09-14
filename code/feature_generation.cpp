#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>
#include <dirent.h>
#include <algorithm>



#include <sstream>
#include <fstream>
#include <cassert>
#include <time.h>
#include <cmath>
#include <numeric>


  

const int tbaseduration = 100000; 
const int slidingt = 100000; 
const int xbaseunit = 18;
const int ybaseunit = 14;
const int colnum = int(346/xbaseunit);
const int rownum = int(260/ybaseunit);
const int points_thr = 400;


using namespace std;

vector<float> Maxofhtm(vector<float> vxin, vector<float> vyin, vector<float> speedin){
    vector<float> ofvector(4);
    vector<vector<float> > hismat(4);
    float angle;
    float temp;
    for (int i = 0; i < 4; ++i){
        hismat[i].resize(3);
    }
    for (int i = 0; i < 4; ++i){
        hismat[i][0] += (2*i+1)*45;
    }
    
    for (int i = 0; i < vxin.size(); ++i){
        angle = (atan2(vyin[i], vxin[i])/M_PI)*180;
        if (angle < 0){
            angle += 360;
        } 
        if (angle >= hismat[3][0] || angle < hismat[0][0]){
            temp = speedin[i];
            hismat[3][1] += temp;
        }
        else{
            for (int j = 0; j < 3; ++j){
                if( angle >= hismat[j][0] && angle < hismat[j+1][0]){
                    temp = speedin[i];
                    hismat[j][1] += temp;
                    break;
                }
            } 
        } 

    }
    for (int k = 0; k < 4; ++k){
        ofvector[k] = hismat[k][1];
    }

    return ofvector;

}



vector<string> getFiles(string cate_dir)
{
	vector<string> files;//存放文件名
 
DIR *dir;
	struct dirent *ptr;
	char base[1000];
 
	if ((dir=opendir(cate_dir.c_str())) == NULL)
        {
		perror("Open dir error...");
                exit(1);
        }
 
	while ((ptr=readdir(dir)) != NULL)
	{
		if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
		        continue;
		else if(ptr->d_type == 8)    ///file

			files.push_back(ptr->d_name);
		else if(ptr->d_type == 10)    ///link file

			continue;
		else if(ptr->d_type == 4)    ///dir
		{
			files.push_back(ptr->d_name);

		}
	}
	closedir(dir);
    sort(files.begin(), files.end());
	return files;
}


vector<vector<float> > readfiles(string loadFeatList)
{

std::ifstream infile_feat(loadFeatList.c_str());
	std::string feature; 
	float feat_onePoint;  
	std::vector<float> lines; 
	std::vector<vector<float> > lines_feat; 
	lines_feat.clear();
 
	while(!infile_feat.eof()) 
	{	
		getline(infile_feat, feature); 
		stringstream stringin(feature); 
		lines.clear();
		while (stringin >> feat_onePoint) {     
			lines.push_back(feat_onePoint);
		}
		lines_feat.push_back(lines); 
	}
	infile_feat.close();
    return lines_feat;
}


vector<float> pmv(vector<float> orie){     
    vector<float> mandv(2);
    float sum = accumulate(orie.begin(), orie.end(), 0.0);
    float mean =  sum / orie.size(); 
    float accum = 0.0;
    for (int i = 0; i < orie.size(); i++){
		accum += (orie[i] - mean)*(orie[i] - mean);
	}
    float square = accum/orie.size(); 
    mandv[0] = mean;
    mandv[1] = square;
    return mandv;
}

vector<float> meavar(vector<vector<float> > mvmat){
    vector<float> mvvec(8);
    vector<float> o1(9);  
    vector<float> o2(9);
    vector<float> o3(9);
    vector<float> o4(9);

    for (int i = 0; i < 9; i ++){
        o1[i] = mvmat[i][0];
        o2[i] = mvmat[i][1];
        o3[i] = mvmat[i][2];
        o4[i] = mvmat[i][3];  
    }
    vector<float> results1 = pmv(o1);
    vector<float> results2 = pmv(o2);
    vector<float> results3 = pmv(o3);
    vector<float> results4 = pmv(o4);

    mvvec[0] = results1[0];
    mvvec[1] = results2[0];
    mvvec[2] = results3[0];
    mvvec[3] = results4[0];
    mvvec[4] = results1[1];
    mvvec[5] = results2[1];
    mvvec[6] = results3[1];
    mvvec[7] = results4[1];

    return mvvec;
}



vector<float> Multicub(float t, int x, int y, int xunit, int yunit, float tunit, vector<vector<float> > &ofmat){

    float tcenter = t + tbaseduration/2;   
    int xcenter = (2*x+1)*xbaseunit/2-1;
    int ycenter = (2*y+1)*ybaseunit/2-1;
    int x0 = xcenter - 3*xunit/2;
    int y0 = ycenter - 3*yunit/2;

    float thr_t1 = tcenter-3*tunit/2;
    float thr_t2 = tcenter-tunit/2;
    float thr_t3 = tcenter+tunit/2;
    float thr_t4 = tcenter+3*tunit/2;

    int thr_x = 3*xunit/2;
    int thr_y = 3*yunit/2;
    float nor = 0;


    vector<float> vx;
    vector<float> vy;
    vector<float> speed;
    vector<float> temp;
    vector<vector<float> > tmp;
    vector<float> xy_vector(8);
    vector<float> xt_vector(8);
    vector<float> yt_vector(8);

    vector<vector<int> > hofevent(19); 
    vector<vector<float> > ofvector(19); 
    for (int i = 0; i < 19; ++i){
        ofvector[i].resize(4);
        }

    vector<vector<vector<int> > > bfofevent;
    bfofevent.resize(3);
        for (int i = 0; i < 3; ++i){
        bfofevent[i].resize(3);
        }
    vector<vector<vector<int> > > cuofevent;
    cuofevent.resize(3);
        for (int i = 0; i < 3; ++i){
        cuofevent[i].resize(3);
        }
    vector<vector<vector<int> > > afofevent;
    afofevent.resize(3);
        for (int i = 0; i < 3; ++i){
        afofevent[i].resize(3);
        }

    for (int i = 0; i < ofmat.size()-1; ++i){ 

        if (ofmat[i][0]<thr_t1 || abs(ofmat[i][1]-xcenter) >= thr_x || abs(ofmat[i][2]-ycenter) >= thr_y){
            continue;
        }   
        else if (ofmat[i][0]>=thr_t1 && ofmat[i][0] < thr_t2){
            
                bfofevent[int((ofmat[i][2]-y0)/yunit)][int((ofmat[i][1]-x0) /xunit)].push_back(i);
            
        }
        else if (ofmat[i][0]>=thr_t2 && ofmat[i][0] < thr_t3){
            
                cuofevent[int((ofmat[i][2]-y0)/yunit)][int((ofmat[i][1]-x0) /xunit)].push_back(i);
            
        }
        else if (ofmat[i][0]>=thr_t3 && ofmat[i][0] <= thr_t4){
            
                afofevent[int((ofmat[i][2]-y0)/yunit)][int((ofmat[i][1]-x0) /xunit)].push_back(i);
            
        }
        else if (ofmat[i][0]> thr_t4){
            break;
        }
    }

    hofevent[0].assign(cuofevent[1][1].begin(), cuofevent[1][1].end());
    hofevent[1].assign(bfofevent[2][1].begin(), bfofevent[2][1].end());
    hofevent[2].assign(bfofevent[1][2].begin(), bfofevent[1][2].end());
    hofevent[3].assign(bfofevent[0][1].begin(), bfofevent[0][1].end());
    hofevent[4].assign(bfofevent[1][0].begin(), bfofevent[1][0].end());
    hofevent[5].assign(bfofevent[1][1].begin(), bfofevent[1][1].end());
    hofevent[6].assign(cuofevent[2][0].begin(), cuofevent[2][0].end());
    hofevent[7].assign(cuofevent[2][1].begin(), cuofevent[2][1].end());
    hofevent[8].assign(cuofevent[2][2].begin(), cuofevent[2][2].end());
    hofevent[9].assign(cuofevent[1][0].begin(), cuofevent[1][0].end());
    hofevent[10].assign(cuofevent[1][2].begin(), cuofevent[1][2].end());
    hofevent[11].assign(cuofevent[0][0].begin(), cuofevent[0][0].end());
    hofevent[12].assign(cuofevent[0][1].begin(), cuofevent[0][1].end());
    hofevent[13].assign(cuofevent[0][2].begin(), cuofevent[0][2].end());
    hofevent[14].assign(afofevent[2][1].begin(), afofevent[2][1].end());
    hofevent[15].assign(afofevent[1][2].begin(), afofevent[1][2].end());
    hofevent[16].assign(afofevent[0][1].begin(), afofevent[0][1].end());
    hofevent[17].assign(afofevent[1][0].begin(), afofevent[1][0].end());
    hofevent[18].assign(afofevent[1][1].begin(), afofevent[1][1].end());


    for (int i = 0; i < 19; ++i){

        vx.clear();
        vy.clear();
        speed.clear();
        if (hofevent[i].size() == 0){
            for (int j = 0; j < 4; j++){
                ofvector[i].at(j) = 0;
            }
        }
        else{
            for (int j = 0; j < hofevent[i].size(); ++j){
                vx.push_back(ofmat[hofevent[i][j]][3]);
                vy.push_back(ofmat[hofevent[i][j]][4]);
                speed.push_back(ofmat[hofevent[i][j]][5]);
            }
            ofvector[i] = Maxofhtm(vx, vy, speed);
        }
    }

    tmp.clear();

    tmp.push_back(ofvector[6]);
    tmp.push_back(ofvector[7]);
    tmp.push_back(ofvector[8]);
    tmp.push_back(ofvector[9]);
    tmp.push_back(ofvector[10]);
    tmp.push_back(ofvector[11]);
    tmp.push_back(ofvector[12]);
    tmp.push_back(ofvector[13]);
    tmp.push_back(ofvector[0]);

    xy_vector = meavar(tmp);
    temp.insert(temp.end(),xy_vector.begin(),xy_vector.end());

    tmp.clear();
    tmp.push_back(ofvector[2]);
    tmp.push_back(ofvector[4]);
    tmp.push_back(ofvector[5]);
    tmp.push_back(ofvector[9]);
    tmp.push_back(ofvector[10]);
    tmp.push_back(ofvector[15]);
    tmp.push_back(ofvector[17]);
    tmp.push_back(ofvector[18]);
    tmp.push_back(ofvector[0]);

    xt_vector = meavar(tmp);
    temp.insert(temp.end(),xt_vector.begin(),xt_vector.end());

    
    tmp.clear();
    tmp.push_back(ofvector[1]);
    tmp.push_back(ofvector[3]);
    tmp.push_back(ofvector[5]);
    tmp.push_back(ofvector[7]);
    tmp.push_back(ofvector[12]);
    tmp.push_back(ofvector[14]);
    tmp.push_back(ofvector[16]);
    tmp.push_back(ofvector[18]);
    tmp.push_back(ofvector[0]);

    yt_vector = meavar(tmp);

    temp.insert(temp.end(),yt_vector.begin(),yt_vector.end());

    return temp;
}


void Neicub(vector<float> sigt, vector<int> sigcubx, vector<int> sigcuby, vector<vector<float> > &ofmat, string &svfn){
  
    vector<float> neit;
    vector<int> neix;
    vector<int> neiy;
    vector<vector<float> > neivector;
    vector<float> temp;
    vector<vector<float> > tmp;
    vector<float> xy_vector;
    vector<float> xt_vector;
    vector<float> yt_vector;
    vector<float> origin(24);
    vector<float> expand(24);
    vector<float> shrink(24);

    int xy_t, xy_x, xy_y;
    int xt_t, xt_x, xt_y;
    int yt_t, yt_x, yt_y;

    for (int i = 0; i < sigt.size(); ++i){

        temp.clear();

        if (sigt[i] < ofmat[0][0] + 2.5*tbaseduration || sigt[i] > ofmat[ofmat.size()-2][0] - 3.5*tbaseduration || sigcubx[i] < 3 
            || sigcubx[i] > colnum-4 || sigcuby[i] < 3 || sigcuby[i] > rownum-4){
            continue;
        }

        origin = Multicub(sigt[i], sigcubx[i], sigcuby[i], xbaseunit, ybaseunit, tbaseduration, ofmat);         
        expand = Multicub(sigt[i], sigcubx[i], sigcuby[i], 2*xbaseunit, 2*ybaseunit, 2*tbaseduration, ofmat);
        shrink = Multicub(sigt[i], sigcubx[i], sigcuby[i], 0.5*xbaseunit, 0.5*ybaseunit, 0.5*tbaseduration, ofmat);

        temp.insert(temp.end(),origin.begin(),origin.end());
        temp.insert(temp.end(),expand.begin(),expand.end());       
        temp.insert(temp.end(),shrink.begin(),shrink.end());


        neivector.push_back(temp);
        neit.push_back(sigt[i] + tbaseduration/2);  
        neix.push_back(sigcubx[i]);
        neiy.push_back(sigcuby[i]);

    } 
     
    for (int i = 0; i < neit.size(); ++i){
        neivector[i].insert(neivector[i].begin(),neiy[i]);
        neivector[i].insert(neivector[i].begin(),neix[i]);
        neivector[i].insert(neivector[i].begin(),neit[i]);
    }


    FILE *fp;
	fp = fopen(svfn.c_str(), "wt");
	for (int i = 0; i < neivector.size(); ++i) {
        for (int j = 0; j < neivector[0].size()-1; ++j){
		fprintf(fp, "%.02f ", neivector[i][j]);
        }
        fprintf(fp, "%.02f\n", neivector[i][71]);
	}
	fclose(fp);
}
void ACuboid(vector<vector<float> > &eventmat, int poithr, vector<vector<float> > &ofmat, string &svfn){

    float starttime;
    int startevent;
    if(ofmat[0][0] <= eventmat[0][0]){
        starttime = eventmat[0][0];
        startevent = 0;  
    }
    else if (ofmat[0][0] > eventmat[0][0])
    {
        starttime = ofmat[0][0];
        for (int i = 0; i < eventmat.size()-1; ++i){
            if (eventmat[i][0] < starttime && eventmat[i+1][0] >= starttime){ 
                startevent = i + 1;
                break;
            }
        }
    
    }
    
    float endtime = ofmat[ofmat.size()-2][0] - tbaseduration; 

    vector<vector<float> > neivector; 
    vector<float> f;
    vector<int> x;
    vector<int> y;

    int thr_x = xbaseunit*colnum;
    int thr_y = ybaseunit*rownum;
    int cuboid[rownum][colnum] = {0};

    while(starttime <= endtime){
        
        for (int i = startevent; i < eventmat.size(); ++i){
            if (eventmat[i][0] <=  starttime + tbaseduration){   
                if (eventmat[i][1] < thr_x && eventmat[i][2] < thr_y){
                    cuboid[int(eventmat[i][2]/ybaseunit)][int(eventmat[i][1]/xbaseunit)] += 1;
                }
            }
            else{
                startevent = i;
                starttime += slidingt;
                break;
            }
        }
        for (int i = 0; i < colnum; ++i){
            for (int j = 0; j < rownum; ++j){   
                if (cuboid[j][i] >= poithr){           
                    y.push_back(j);
                    x.push_back(i);
                    f.push_back(starttime - slidingt);  
                                 
                }
            }
        } 
        memset(cuboid,0,sizeof(cuboid));                
    }
    Neicub(f, x, y, ofmat , svfn);

}


int main()
{
   
    string eventspath = "/media/lpg/My Passport/AED/anomaly_dataset/walking/events/nor/";
    string ofpath = "/media/lpg/My Passport/AED/anomaly_dataset/walking/fixof/nor/";
    string savepath = "/media/lpg/My Passport/AED/anomaly_dataset/walking/result/nor_400/";
    


    vector<string> files=getFiles(ofpath);
    for (int i=0; i<files.size(); i++)
    {   
    	string offn = ofpath + files[i];
        string enfn = eventspath + files[i];
        string svfn = savepath + files[i];
        vector<vector<float> > ofmat = readfiles(offn);
        vector<vector<float> > eventmat = readfiles(enfn);
        ACuboid(eventmat,points_thr, ofmat , svfn);

    }
    return 0;
    
}



