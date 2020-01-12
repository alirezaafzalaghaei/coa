#include <iostream>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <vector>
#include <time.h>
#include <omp.h>
#include <unistd.h>
#include <chrono>

using namespace std;
using namespace arma;
using namespace mlpack::kmeans;

#define PI 3.14159265357989

class Cuckoo {

public:
    size_t number_of_eggs;
    rowvec center;
    rowvec egg_laying_radiuses;
    colvec profit_values;
    mat new_pos_4_Egg;
};

class Cluster {
public:
    mat positions;
    colvec profits;
    // rowvec center;

};

class COA {
public:
    unsigned int npar;
    size_t max_iter, n_clusters;
    size_t min_num_of_eggs, max_num_of_eggs;
    size_t num_of_cuckoos, max_num_of_cuckoos;

    double var_lo, var_hi;
    double motion_coeff, radius_coeff;
    double accuracy, cuckoo_pop_variance;
    bool verbose;
    vector < double > cost_vector;
    function<double(rowvec)> cost_func;

    colvec cost_function(mat habitats) {
        size_t n = habitats.n_rows;
        colvec result(n);

        #pragma omp parallel for if (n > 10) default(none) shared(habitats, result, n) schedule(auto)
        for (size_t i = 0; i < n; i++)
            result(i) = cost_func(habitats.row(i));

        return result;
    }

    rowvec run() {
        size_t iteration = 0, sumOfEggs;
        double globalMaxProfit;
        colvec tmp, tmpProfits;
        rowvec globalBestCuckoo, tmp2, bestCuckooCenter;

        vector < Cuckoo > CuckooPop(num_of_cuckoos);


        for (size_t i = 0; i < num_of_cuckoos; i++)
            CuckooPop[i].center = (var_hi - var_lo) * randu < rowvec > (npar) + var_lo;

        globalBestCuckoo = CuckooPop[0].center;

        tmp = -1 * cost_function(mat(globalBestCuckoo));
        globalMaxProfit = tmp(0);

        while (++iteration <= max_iter && -globalMaxProfit > accuracy) {
            unsigned int CuckooPopSize = CuckooPop.size();
            sumOfEggs = 0;
            for (size_t i = 0; i < CuckooPopSize; i++) {
                CuckooPop[i].number_of_eggs = min_num_of_eggs + (int)((max_num_of_eggs - min_num_of_eggs) * randu());
                sumOfEggs += CuckooPop[i].number_of_eggs;
            }
            #pragma omp parallel for if (CuckooPopSize > 100) schedule(auto) num_threads(2)
            for (size_t i = 0; i < CuckooPopSize; i++) {
                double eggLayingRadius = (float) CuckooPop[i].number_of_eggs / sumOfEggs * (radius_coeff * (var_hi - var_lo));
                rowvec tmpx = randu < rowvec > (CuckooPop[i].number_of_eggs);
                CuckooPop[i].egg_laying_radiuses = eggLayingRadius * tmpx;

                rowvec angles = linspace < rowvec > (0, 2 * PI, CuckooPop[i].number_of_eggs);
                mat newpos(CuckooPop[i].number_of_eggs, npar);

                for (size_t cnt = 0; cnt < CuckooPop[i].number_of_eggs; cnt++) {
                    irowvec signs = 2 * randi<irowvec>(npar,distr_param(0,1)) - 1;

                    rowvec tmp2 = CuckooPop[i].center + (signs * (CuckooPop[i].egg_laying_radiuses(cnt) * cos(angles(cnt)) + CuckooPop[i].egg_laying_radiuses(cnt) * sin(angles(cnt))));

                    tmp2(arma::find(tmp2 > var_hi)).fill(var_hi);
                    tmp2(arma::find(tmp2 < var_lo)).fill(var_lo);

                    newpos.row(cnt) = tmp2;
                }

                CuckooPop[i].new_pos_4_Egg = newpos;
            }

            mat allPositions(sumOfEggs + num_of_cuckoos, npar);
            int k = 0;
            #pragma omp parallel for if (CuckooPopSize > 100) shared(k) schedule(auto) num_threads(2)
            for (size_t i = 0; i < CuckooPopSize; i++) {
                allPositions.row(k++) = CuckooPop[i].center;
                for (size_t j = 0; j < CuckooPop[i].number_of_eggs; j++)
                    allPositions.row(k++) = CuckooPop[i].new_pos_4_Egg.row(j);
            }

            tmpProfits = -1 * cost_function(allPositions);
            int cur = 0;
            for (size_t i = 0; i < CuckooPopSize; i++) {
                CuckooPop[i].profit_values = tmpProfits.subvec(cur, cur + CuckooPop[i].number_of_eggs);
                cur += CuckooPop[i].number_of_eggs + 1;
            }

            uvec sortedIndex = sort_index(tmpProfits, "descend");
            bestCuckooCenter = allPositions.row(sortedIndex(0));

            tmpProfits = tmpProfits(sortedIndex);
            allPositions = allPositions.rows(sortedIndex);

            if (num_of_cuckoos > max_num_of_cuckoos) {
                allPositions.shed_rows(max_num_of_cuckoos + 1, allPositions.n_rows - 1);
                tmpProfits.shed_rows(max_num_of_cuckoos + 1, tmpProfits.n_rows - 1);

                CuckooPop = vector < Cuckoo > (max_num_of_cuckoos);
                for (size_t i = 0; i < max_num_of_cuckoos; i++) {
                    CuckooPop[i].center = allPositions.row(i);
                    CuckooPop[i].new_pos_4_Egg = allPositions.row(i);
                    CuckooPop[i].profit_values = tmpProfits.row(i);
                }
                num_of_cuckoos = max_num_of_cuckoos;
                CuckooPopSize = CuckooPop.size();
            }

            rowvec current_best_cuckoo = allPositions.row(0);
            double current_max_profit = tmpProfits(0);

            if (current_max_profit > globalMaxProfit) {
                globalBestCuckoo = current_best_cuckoo;
                globalMaxProfit = current_max_profit;
            }

            cost_vector.push_back(-globalMaxProfit);

            if (accu(var (allPositions)) < cuckoo_pop_variance) {
                cout << "cuckoo_pop_variance reached, so breaking...\n";
                break;
            }

            Row < size_t > clusterNumbers;
            KMeans < > kmeans(20); // suppose no empty cluster
            kmeans.Cluster(allPositions.t(), n_clusters, clusterNumbers);
            vector < Cluster > clusters(n_clusters);
            rowvec f_mean(n_clusters);

            for (size_t i = 0; i < n_clusters; ++i) {
                uvec in_ith_cluster = arma::find(clusterNumbers == i);

                clusters[i].positions = allPositions.rows(in_ith_cluster);
                clusters[i].profits = tmpProfits.rows(in_ith_cluster);

                // clusters[i].center = mean(clusters[i].positions, 0);
                // clusterNumbers.print("cluster numbers = ");
                f_mean(i) = mean(clusters[i].profits);

            }

            uvec sorted_f_mean = sort_index(f_mean, "descend");
            size_t indexOfBestCluster = sorted_f_mean(0);

            size_t indexOfBestEggPosition = clusters[indexOfBestCluster].profits.index_max();
            rowvec goalPoint = clusters[indexOfBestCluster].positions.row(indexOfBestEggPosition);

            num_of_cuckoos = 0;
            for (size_t i = 0; i < n_clusters; ++i) {
                mat tmpPositions = clusters[i].positions;
                int n_cuckoos = tmpPositions.n_rows;
                tmpPositions += motion_coeff * randu < mat > (n_cuckoos, npar) % (goalPoint - tmpPositions.each_row());

                tmpPositions(arma::find(tmpPositions > var_hi)).fill(var_hi);
                tmpPositions(arma::find(tmpPositions < var_lo)).fill(var_lo);

                clusters[i].positions = tmpPositions;
                // clusters[i].center = mean(tmpPositions);
                num_of_cuckoos += n_cuckoos;
            }
            k = 0;
            CuckooPop = vector < Cuckoo > (num_of_cuckoos);
            for (size_t i = 0; i < n_clusters; ++i) {
                mat tmpPositions = clusters[i].positions;
                for (size_t j = 0; j < tmpPositions.n_rows; ++j)
                    CuckooPop[k++].center = tmpPositions.row(j);
            }
            CuckooPopSize = CuckooPop.size();
            CuckooPop[--k].center = globalBestCuckoo;
            tmp2 = globalBestCuckoo % randu < rowvec > (npar);

            tmp2(arma::find(tmp2 > var_hi)).fill(var_hi);
            tmp2(arma::find(tmp2 < var_lo)).fill(var_lo);

            CuckooPop[--k].center = tmp2;
            if (verbose)
                printf("%04zuth iteration, cost = %10.4e\n", iteration, -globalMaxProfit);

        }

        return globalBestCuckoo;
    }

    COA(function<double(rowvec)> f, unsigned int n_par, double var_lo,double var_hi) {
        this->npar = n_par;
        this->var_lo = var_lo;
        this->var_hi = var_hi;
        this->cost_func = f;

        this->num_of_cuckoos = 5;
        this->max_num_of_cuckoos = 10;
        this->max_iter = 100;
        this->min_num_of_eggs = 2;
        this->max_num_of_eggs = 4;
        this->radius_coeff = 1;
        this->motion_coeff = 1;
        this->n_clusters = 1;
        this->cuckoo_pop_variance = 1e-15;
        this->accuracy = 1e-15;
        this->verbose = true;
    }

};

double ackley2(rowvec x) {
    double result;
    size_t n = x.n_cols;
    double s1=0, s2=0;
    for (size_t j=0; j<n; j++)
    {
        s1+=pow(x(j),2);
        s2+=cos(2*PI*x(j));
    }

    result = -20 * exp(-.2 * sqrt(1.0/n * s1)) - exp(1.0/n * s2)+ 20 + exp(1);

    return result;
}

double sphere(rowvec x)
{
    return accu(pow(x, 2));
}


// solve a simple integral equation
double residual(rowvec p)
{    
    double a=0, b=1;
    int n = 501;
    colvec x = linspace<colvec>(a, b, n);
    colvec fx = exp(x) - x;
    colvec u = polyval(p, x);
    colvec u_hat(n);
    colvec t = linspace<colvec>(a, b, n);

    for(int i = 0 ; i<n;i++)
    {
        colvec ku = x(i) * t % polyval(p,t);
        mat int_ku = trapz(t, ku);
        u_hat(i) = fx(i) + int_ku(0,0);
    }

    return accu(pow(u - u_hat, 2));
}


int main() {

    unsigned int npar = 7;
    double var_lo = 0, var_hi = 1;

    COA coa(residual, npar, var_lo, var_hi);

    coa.num_of_cuckoos = 25;
    coa.max_num_of_cuckoos = 50;
    coa.motion_coeff = 20;
    coa.radius_coeff = 15;

    coa.max_iter = 30;
    coa.cuckoo_pop_variance = 0;
    coa.accuracy = 0;
    coa.verbose = false;
    auto start = chrono::high_resolution_clock::now();
    rowvec c = coa.run();
    auto finish = chrono::high_resolution_clock::now();
    chrono::duration < double > elapsed = finish - start;
    cout << (int)(elapsed.count() * 1000000) << endl;

    //cout << "Last cost value " << coa.cost_vector.back() << endl;
    //cout << c;
    return 0;
}
