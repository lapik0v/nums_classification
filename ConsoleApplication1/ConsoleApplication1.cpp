#include <fstream>
#include <random>
#include <Windows.h>
#include <iomanip>
#include <iostream>
#include <gdiplus.h>
#pragma comment(lib,"gdiplus.lib")
using namespace Gdiplus;
using namespace std;

struct neuron {
	double value;
	double error;
	void act() {
		value = (1 / (1 + pow(2.71828, -value)));
	}
};

struct data_one {
	double info[4096];
	char rresult;
};

struct network {
	int layers;
	neuron** neurons;
	double*** weights; // 1 разрядность - слой, 2 - номер нейрона, 3 - номер связи со след.слоем
	int* size; // количество нейронов в каждом слою

	double sigm_pro(double x) {
		/* производная сигмоиды */
		if ((fabs(x - 1) < 1e-9) || (fabs(x) < 1e-9)) return 0.0;
		double res = x * (1.0 - x);
		return res;
	}

	void setLayersNotStudy(int n, int* p, string filename) {
		/* создание слоев, заполнение натренированными весами */
		ifstream fin;
		fin.open(filename);
		layers = n;
		neurons = new neuron * [n];
		weights = new double** [n - 1];
		size = new int[n];
		for (int i = 0; i < n; i++) {
			size[i] = p[i];
			neurons[i] = new neuron[p[i]];
			if (i < n - 1) {
				weights[i] = new double* [p[i]];
				for (int j = 0; j < p[i]; j++) {
					weights[i][j] = new double[p[i + 1]];
					for (int k = 0; k < p[i + 1]; k++) {
						fin >> weights[i][j][k];
					}
				}
			}
		}
	}

	void setLayers(int n, int* p) {
		/* генерация рандомных весов */
		layers = n;
		neurons = new neuron * [n];
		weights = new double** [n - 1];
		size = new int[n];
		for (int i = 0; i < n; i++) {
			size[i] = p[i];
			neurons[i] = new neuron[p[i]];
			if (i < n - 1) {
				weights[i] = new double* [p[i]];
				for (int j = 0; j < p[i]; j++) {
					weights[i][j] = new double[p[i + 1]];
					for (int k = 0; k < p[i + 1]; k++) {
						weights[i][j][k] = ((rand() % 100)) * 0.01 / size[i];
					}
				}
			}
		}
	}

	void set_input(double p[]) {
		/* ввод данных 1 слоя */
		for (int i = 0; i < size[0]; i++) {
			neurons[0][i].value = p[i];
		}
	}

	void LayersCleaner(int LayerNumber, int start, int stop) {
		/* чистка слоев */
		for (int i = start; i < stop; i++) {
			neurons[LayerNumber][i].value = 0;
		}
	}

	void ForwardFeeder(int LayerNumber, int start, int stop) {
		for (int j = start; j < stop; j++) {
			for (int k = 0; k < size[LayerNumber - 1]; k++) {
				neurons[LayerNumber][j].value += neurons[LayerNumber - 1][k].value * weights[LayerNumber - 1][k][j];
			}
			neurons[LayerNumber][j].act();
		}
	}

	double ForwardFeed() {
		/* перемножение весов на входы */
		setlocale(LC_ALL, "ru");
		for (int i = 1; i < layers; i++) {
			LayersCleaner(i, 0, size[i]);
			ForwardFeeder(i, 0, size[i]);
		}
		double max = 0;
		double prediction = 0;
		for (int i = 0; i < size[layers - 1]; i++) {
			if (neurons[layers - 1][i].value > max) {
				max = neurons[layers - 1][i].value;
				prediction = i;
			}
		}
		return prediction;
	}

	double ShowResults() {
		setlocale(LC_ALL, "ru");
		for (int i = 1; i < layers; i++) {
			LayersCleaner(i, 0, size[i]);
			ForwardFeeder(i, 0, size[i]);
		}
		double max = 0;
		double prediction = 0;
		for (int i = 0; i < size[layers - 1]; i++) {
			cout << char(i + 48) << " : " << neurons[layers - 1][i].value << endl;
			if (neurons[layers - 1][i].value > max) {
				max = neurons[layers - 1][i].value;
				prediction = i;
			}
		}
		return prediction;
	}

	void WeightsUpdater(int start, int stop, int LayerNum, int lr) {
		/* обновляем веса */
		int i = LayerNum;
		for (int j = start; j < stop; j++) {
			for (int k = 0; k < size[i + 1]; k++) {
				weights[i][j][k] += lr * neurons[i + 1][k].error * sigm_pro(neurons[i + 1][k].value) * neurons[i][j].value;
			}
		}
	}

	void BackPropogation(double prediction, double rresult, double lr) {
		/* обратное распространение ошибки */
		for (int i = layers - 1; i > 0; i--) {
			if (i == layers - 1) {
				for (int j = 0; j < size[i]; j++) {
					if (j != int(rresult)) {
						neurons[i][j].error = -pow((neurons[i][j].value), 2);
					}
					else {
						neurons[i][j].error = pow(1.0 - neurons[i][j].value, 2);
					}
				}
			}
			else {
				for (int j = 0; j < size[i]; j++) {
					double error = 0.0;
					for (int k = 0; k < size[i + 1]; k++) {
						error += neurons[i + 1][k].error * weights[i][j][k];
					}
					neurons[i][j].error = error;
				}
			}
		}
		for (int i = 0; i < layers - 1; i++) {
			for (int j = 0; j < size[i]; j++) {
				for (int k = 0; k < size[i + 1]; k++) {
					weights[i][j][k] += lr * neurons[i + 1][k].error * sigm_pro(neurons[i + 1][k].value) * neurons[i][j].value;
				}
			}

		}
	}

	bool SaveWeights() {
		/* перезапись весов */
		ofstream fout;
		fout.open("weights.txt");
		for (int i = 0; i < layers; i++) {
			if (i < layers - 1) {
				for (int j = 0; j < size[i]; j++) {
					for (int k = 0; k < size[i + 1]; k++) {
						fout << weights[i][j][k] << " ";
					}
				}
			}
		}
		fout.close();
		return 1;
	}

	void GetTest()	{
		/* перевод изображения в матрицу */
		Gdiplus::GdiplusStartupInput input;
		Gdiplus::GdiplusStartupOutput output;
		ULONG_PTR token;
		Gdiplus::GdiplusStartup(&token, &input, &output);

		Gdiplus::Color color;
		Gdiplus::Bitmap* bitmap;

		bitmap = new Gdiplus::Bitmap(L"image.png");
		ofstream ofs("test.txt");
		fstream fout("test.txt");
		fout << fixed << showpoint << setprecision(3);
		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < 64; j++) {
				bitmap->GetPixel(j, i, &color);
				fout << (1.0 - ((unsigned)color.GetRed() / 255.0)) << " ";
			}
			fout << endl;
		}
		fout.close();
	}
};

int main() {
	setlocale(LC_ALL, "Russian");
	ifstream fin;
	ofstream fout;
	const int l = 4; // количество слоев 
	const int input_l = 4096; // количество нейронов на входе (64х64)
	int size[l] = { input_l, 256, 64,   10 }; // массив количества нейронов на слоях
	network nn;

	double input[input_l]; // ввод для обучающей выборки
	char rresult; // правильный результат
	double result;
	double ra = 0;
	int maxra = 0;
	int maxraepoch = 0;
	const int n = 150; // размер обучающей выборки
	bool to_study = 0;
	cout << "Обучить по lib.txt (1/0)?" << endl;
	cin >> to_study;

	data_one* data = new data_one[n]; // массив с обучающим датасетом размера n

	if (to_study) {
		fin.open("lib.txt");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < input_l; j++) {
				fin >> data[i].info[j]; // запись матрицы изображения
			}
			fin >> data[i].rresult; // запись правильной цифры
			data[i].rresult -= 48;
		}

		nn.setLayers(l, size); // создаем слои, веса

		for (int e = 0; ra / n * 100 < 100; e++) {
			/* цикл эпох */
			ra = 0; // количество правильных ответов
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < 4096; j++) {
					input[j] = data[i].info[j]; // запись матрицы изображения
				}
				rresult = data[i].rresult; // запись правильной цифры
				nn.set_input(input); // подаем нейросети на 1 слой входные данные

				result = nn.ForwardFeed();

				if (result == rresult) {
					cout << "Угадал цифру " << char(rresult + 48) << "\t\t\t****" << endl;
					ra++;
				}
				else {
					cout << "Результат " << result << " неверный!\n";
					cout << "Не угадал цифру " << char(rresult + 48) << "\n";
					nn.BackPropogation(result, rresult, 0.5);
				}
			}
			if (ra > maxra) {
				maxra = ra;
				maxraepoch = e;
			}
			if (maxraepoch < e - 250) {
				maxra = 0;
			}
			cout << "Ошибка: " << 100 - (ra / n * 100) << "% \t Мин.ошибка: " << 100 - (double(maxra) / n * 100) << "(на эпохе № " << maxraepoch << ")" << endl;
		}
		if (nn.SaveWeights()) {
			cout << "Веса сохранены!" << endl;
		}
	}
	else {
		nn.setLayersNotStudy(l, size, "weights.txt");
	}

	cout << "Проверить image.png (1/0)?" << endl;
	bool to_start_test = 0;
	cin >> to_start_test;
	char right_res;
	if (to_start_test) {
		nn.GetTest();
		fin.open("test.txt");
		for (int i = 0; i < input_l; i++) {
			fin >> input[i];
		}
		nn.set_input(input);
		result = nn.ShowResults();
		cout << "Должно быть это цифра " << char(result + 48) << "\n\n";
		cout << "А какая это цифра на самом деле?..." << endl;
		cin >> right_res;
		if (right_res != result + 48) {
			cout << "Исправляем веса..." << endl;
			nn.BackPropogation(result, right_res - 48, 0.15);
			nn.SaveWeights();
		}
	}
	return 0;
}