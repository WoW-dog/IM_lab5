#include <iostream>
#include <vector>
#include <cmath>
#include <Windows.h>
#include <fstream>
#include <iomanip>
#include "Eigen\Eigen\Dense"

struct DataPoint {
    Eigen::VectorXd factors;
    double target;
    DataPoint(const Eigen::VectorXd& f, double t) : factors(f), target(t) {}
};

// Функция для построения линейной многофакторной модели методом наименьших квадратов
Eigen::VectorXd buildModel(std::vector<DataPoint>& data) {
    int numFactors = data[0].factors.size();
    int numDataPoints = data.size();

    Eigen::MatrixXd X(numDataPoints, numFactors + 1);
    Eigen::VectorXd y(numDataPoints);

    for (int i = 0; i < numDataPoints; ++i) {
        X(i, 0) = 1.0;
        X.block(i, 1, 1, numFactors) = data[i].factors.transpose();
        y(i) = data[i].target;
    }

    Eigen::VectorXd coefficients = (X.transpose() * X).ldlt().solve(X.transpose() * y);
    return coefficients;
}

// Функция для предсказания значения целевой переменной
double predict(const Eigen::VectorXd& coefficients, Eigen::VectorXd& factors) {
    Eigen::VectorXd extendedFactors(coefficients.size());
    extendedFactors(0) = 1.0;
    extendedFactors.segment(1, factors.size()) = factors;
    return extendedFactors.dot(coefficients);
}

void readData(std::string pathToFile, std::vector<DataPoint>& data, int &n, int &m) {
    std::ifstream inputFile(pathToFile);
    if (inputFile.is_open()) {
        inputFile >> n >> m;
        Eigen::VectorXd tmp(m);
        double tmpCell;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                inputFile >> tmpCell;
                tmp(j) = tmpCell;
            }
            inputFile >> tmpCell;
            data.push_back(DataPoint(tmp, tmpCell));
        }
        inputFile.close();
    }
}

void predictData(std::string pathToFile, std::vector<DataPoint>& data, Eigen::VectorXd& coefficients, int& n) {
    std::ifstream inputFile(pathToFile);
    if (inputFile.is_open()) {
        int m;
        inputFile >> n >> m;
        Eigen::VectorXd tmp(m);
        double tmpCell;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                inputFile >> tmpCell;
                tmp(j) = tmpCell;
            }
            tmpCell = predict(coefficients, tmp);
            data.push_back(DataPoint(tmp, tmpCell));
        }
        inputFile.close();
    }
}

double sigmaE2(std::vector<DataPoint>& data, Eigen::VectorXd& coefficients, int n, int m) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (data[i].target - predict(coefficients, data[i].factors)) * (data[i].target - predict(coefficients, data[i].factors));
    }
    return sum / (double)(n - m - 1);
}

double sigmaA2(std::vector<DataPoint>& data, int n, int m) {
    double avg = 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        avg += data[i].target;
    }
    avg = avg / (double)(n);
    for (int i = 0; i < n; i++) {
        sum += (data[i].target - avg) * (data[i].target - avg);
    }
    return sum / (double)(n - 1);
}
// Функция для вычисления коэффициента корреляции между факторами и откликом
double D(std::vector<DataPoint>& data, Eigen::VectorXd& coefficients, int n, int m) {
    return 1.0 - sigmaE2(data, coefficients, n, m) / sigmaA2(data, n, m);
}

// Построение матрицы корреляции
std::vector<Eigen::VectorXd> matrixCorrelation(std::vector<DataPoint>& data, int n, int m) {
    std::cout << std::fixed << std::showpoint << std::setprecision(2);
    double sum1, sum21, sum22, avg_i, avg_j;
    std::vector<Eigen::VectorXd> matrix;
    Eigen::VectorXd tmp(m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            sum1 = 0.0;
            sum21 = 0.0;
            sum22 = 0.0;
            avg_i = 0.0;
            avg_j = 0.0;
            for (int k = 0; k < n; k++) {
                avg_i += data[k].factors(i);
                avg_j += data[k].factors(j);
            }
            avg_i /= n;
            avg_j /= n;
            for (int k = 0; k < n; k++) {
                sum1 += (data[k].factors(i) - avg_i) * (data[k].factors(j) - avg_j);
                sum21 += (data[k].factors(i) - avg_i) * (data[k].factors(i) - avg_i);
                sum22 += (data[k].factors(j) - avg_j) * (data[k].factors(j) - avg_j);
            }
            tmp(j) = sum1 / (std::sqrt(sum21 * sum22));
        }
        matrix.push_back(tmp);
    }

    std::cout << "  " << "\t";
    for (int i = 0; i < m; i++) {
        std::cout << "x" << i + 1 << "\t";
    }
    std::cout << "\n";
    for (int i = 0; i < m; i++) {
        std::cout << "x" << i + 1 << "\t";
        for (int j = 0; j < m; j++) {
            std::cout << matrix[i](j) << "\t";
        }
        std::cout << "\n";
    }
    return matrix;
}

// Оценка коэффициента детерминации на статистическую значимость 
double calculateStatisticalSignificance(std::vector<DataPoint>& data, Eigen::VectorXd& coefficients, int n, int m) {
    return (D(data, coefficients, n, m) * (n - m - 1)) / (m - D(data, coefficients, n, m));
}

// Вычисление ошибки
double calculateError(std::vector<DataPoint>& data, Eigen::VectorXd& coefficients, int n, int m) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (data[i].target - predict(coefficients, data[i].factors))/ (predict(coefficients, data[i].factors));
    }
    return 100.0 * sum / (double)(n);
}

void writeData(std::string pathToFile, std::vector<DataPoint>& data, std::vector<DataPoint>& newFactors, std::vector<Eigen::VectorXd> matrixCorr, Eigen::VectorXd& coefficients, int n, int n2, int m) {
    std::ofstream outputFile(pathToFile);
    outputFile << std::fixed << std::showpoint << std::setprecision(2);
    if (outputFile.is_open()) {
        outputFile << "Коэффициенты модели:\n";
        outputFile << coefficients << "\n\n";

        outputFile << "Матрица корреляции факторов между собой:\n";
        outputFile << "  " << "\t";
        for (int i = 0; i < m; i++) {
            outputFile << "x" << i + 1 << "\t";
        }
        outputFile << "\n";
        for (int i = 0; i < m; i++) {
            outputFile << "x" << i + 1 << "\t";
            for (int j = 0; j < m; j++) {
                outputFile << matrixCorr[i](j) << "\t";
            }
            outputFile << "\n";
        }
        outputFile << "\n";

        outputFile << "Коэффициент корреляции между факторами и откликом: " << D(data, coefficients, n, m) << "\n\n";

        outputFile << "Оценка коэффициента детерминации на статистическую значимость:\n" << "F = " << calculateStatisticalSignificance(data, coefficients, n, m) << "\n";
        if (calculateStatisticalSignificance(data, coefficients, n, m) > 0.05) {
            outputFile << "Модель адекватна";
        }
        else {
            outputFile << "Модель неадекватна";
        }
        outputFile << "\n\n";

        outputFile << "Вычисление ошибки:\nE = " << calculateError(data, coefficients, n, m) << "%\n\n";


        outputFile << "Предсказания для новых факторов:\n";
        for (int i = 0; i < m; i++) {
            outputFile << "x" << i + 1 << "\t\t";
        }
        outputFile << "y" << "\n";

        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < m; j++) {
                outputFile << newFactors[i].factors(j) << "\t";
            }
            outputFile << newFactors[i].target << "\n";
        }

        outputFile.close();
        std::cout << "Результаты успешно записаны в файл " << pathToFile << "\n\n";
    }
}

int main() {
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    std::vector<DataPoint> data;
    std::vector<DataPoint> newFactors;
    std::vector<Eigen::VectorXd> matrixCorr;
    std::string pathToInputFile = "input.txt";
    std::string pathToNewFactors = "newFactors.txt";
    std::string pathToOutputFile = "output.txt";
    int n, m, n2;

    readData(pathToInputFile, data, n, m);
    Eigen::VectorXd coefficients = buildModel(data);

    std::cout << "Коэффициенты модели:\n";
    std::cout << coefficients << "\n\n";
    predictData(pathToNewFactors, newFactors, coefficients, n2);

    std::cout << "Матрица корреляции факторов между собой:\n";
    matrixCorr = matrixCorrelation(data, n, m);
    std::cout << "\n";

    std::cout << "Коэффициент корреляции между факторами и откликом: " << D(data, coefficients, n, m) << "\n\n";
    
    std::cout << "Оценка коэффициента детерминации на статистическую значимость:\n" << "F = " << calculateStatisticalSignificance(data, coefficients, n, m) << "\n";
    if (calculateStatisticalSignificance(data, coefficients, n, m) > 0.05) {
        std::cout << "Модель адекватна";
    }
    else {
        std::cout << "Модель неадекватна";
    }
    std::cout << "\n\n";

    std::cout << "Вычисление ошибки:\nE = " << calculateError(data, coefficients, n, m) << "%\n\n";

    std::cout << "Предсказания для новых факторов:\n";
    for (int i = 0; i < m; i++) {
        std::cout << "x" << i + 1 << "\t";
    }
    std::cout << "y" << "\n";

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << newFactors[i].factors(j) << "\t";
        }
        std::cout << newFactors[i].target << "\n";
    }
    std::cout << "\n";

    writeData(pathToOutputFile, data, newFactors, matrixCorr, coefficients, n, n2, m);
}
