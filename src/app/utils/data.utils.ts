import {HttpClient} from "@angular/common/http";
import {Injectable} from "@angular/core";
import {Observable, Subject} from "rxjs";
import {MathUtils, Matrix} from "./math.utils";

@Injectable()
export class DataUtils {

  private _trainingSamples: Array<number[]>;
  private _validationSamples: Array<number[]>;
  private _testingSamples: Array<number[]>;

  private _expectedOutput: Array<number[]>;

  private configuration = {
    epochs: 1000,
    training: 70,
    validation: 15,
    testing: 15,
    inputCount: 0,
    contextCount: 0,
    hiddenCount: 0,
    outputCount: 0
  };

  private _inputWeights: Matrix = null;
  private _hiddenWeights: Matrix = null;

  public loading: boolean = false;

  public training: boolean = false;
  public trained: boolean = false;

  constructor(
    private http: HttpClient
  ) {
  }

  getProgress(epoch: number): string {
    return `${Math.min(100, Math.round(epoch / this.configuration.epochs) * 100)}%`
  }

  get trainingSamples(): Array<number[]> {
    return this._trainingSamples;
  }

  get validationSamples(): Array<number[]> {
    return this._validationSamples;
  }

  get testingSamples(): Array<number[]> {
    return this._testingSamples;
  }

  get inputWeights(): Matrix {
    if (!this._inputWeights) {
      let matrix = new Matrix(
        this.configuration.inputCount + this.configuration.contextCount,
        this.configuration.hiddenCount
      );
      for (let i = 0; i < this.configuration.inputCount + this.configuration.hiddenCount; i++) {
        for (let j = 0; j < this.configuration.hiddenCount; j++) {
          matrix.set(i, j, Math.random());
        }
      }
      this._inputWeights = matrix;
    }
    return this._inputWeights;
  }

  get hiddenWeights(): Matrix {
    if (!this._hiddenWeights) {
      let matrix = new Matrix(
        this.configuration.hiddenCount,
        this.configuration.outputCount
      );
      for (let i = 0; i < this.configuration.hiddenCount; i++) {
        for (let j = 0; j < this.configuration.outputCount; j++) {
          matrix.set(i, j, Math.random());
        }
      }
      this._hiddenWeights = matrix;
    }
    return this._hiddenWeights;
  }

  public configure(epochs: number = 1000, training: number = 70, validation: number = 15) {
    if (epochs <= 0) {
      throw new Error("Epochs count must be greater than 0.");
    }
    if (training <= 0 || validation <= 0) {
      throw new Error("Samples size must be greater than 0.");
    }
    if (training + validation > 99) {
      throw new Error("Training, testing and validation percents must summary equals 100.");
    }
    this.configuration.training = training;
    this.configuration.validation = validation;
    this.configuration.testing = 100 - training - validation;
  }

  /**
   * Основной метод: запускает обучения, а затем тестирование и выводит результаты
   */
  public run() {
    this.train();
  }

  public currentEpoch;

  public currentEpochSubject = new Subject();

  public currentEpoch$ = this.currentEpochSubject.asObservable();

  public train() {
    this.training = true;
    const epochs = this.configuration.epochs;
    if (epochs > 0) {
      const inputCount = this.configuration.inputCount;
      const contextCount = this.configuration.contextCount;

      //iterate by epochs
      for (this.currentEpoch = 0; this.currentEpoch < epochs; this.currentEpoch++) {
        this.currentEpochSubject.next(this.currentEpoch);
        let x: number[] = Array(inputCount + contextCount).fill(0);
        // console.log(inputCount, " ", contextCount);
        // console.log("X = ", x);
        //iterate by training samples
        let actualOutput = [];
        let e = [];
        for (let index = 0; index < this.trainingSamples.length; index++) {
          let currentSample: number[] = this.trainingSamples[index]; // vector, [x1, x2, x3, x4]
          for (let i = 0; i < inputCount; i++) {
            x[i] = currentSample[i];
          }
          let w: Matrix = this.inputWeights;

          // console.log("W: ", w.toString());

          //iterate by hidden neurons
          let hiddenWeightSums = [];
          for (let i = 0; i < this.configuration.hiddenCount; i++) {
            /*
            на вход h[i] приходит [x1, x2, x3, x4, h1, h2, h3] и веса [[w1, w2, w3], [...]]
             */
            let sum = 0;
            for (let j = 0; j < this.configuration.inputCount + this.configuration.contextCount; j++) {
              // w[j][i] * x[j], взвешенная сумма i-го скрытого нейрона
              for (let k = 0; k < x.length; k++) {
                sum += w.get(j, i) * x[k];
              }
            }
            hiddenWeightSums[i] = sum;
            x[inputCount + i] = MathUtils.sigmoid(hiddenWeightSums[i]);
          }

          //iterate by output neurons
          w = this.hiddenWeights;
          let outputWeightSums = [];
          let currentOutput = [];
          for (let s = 0; s < this.configuration.outputCount; s++) {
            /*
            на вход o[s] приходит [h1, h2, h3] и веса
             */
            let sum = 0;
            for (let i = 0; i < this.configuration.hiddenCount; i++) {
              // w[i][s] * h[i], взвешенная сумма нейрона
              for (let j = 0; j < this.configuration.hiddenCount; j++) {
                sum += w.get(i, s) * x[inputCount + j];
                // console.log(x);
                // console.log(x.length, " ", j, " ", inputCount + j);
                // console.log(w.get(i, s), " ", x[inputCount + j])
              }
            }
            outputWeightSums[s] = sum;
            currentOutput[s] = MathUtils.sigmoid(outputWeightSums[s]);
          }
          actualOutput = [...actualOutput, ...currentOutput];

          //error
          let currentErrors = [];
          let resultError = 0;
          for (let i = 0; i < this.configuration.outputCount; i++) {
            let error = this._expectedOutput[index][i] - currentOutput[i];
            currentErrors.push(error);
            resultError += Math.pow(error, 2);
          }
          resultError = resultError / 2;

          if (resultError) {
            e.push(resultError);
          }

          // корректировка весов
          this.updateInputWeights(this.currentEpoch, x, hiddenWeightSums, outputWeightSums, currentErrors);
          this.updateHiddenWeights(hiddenWeightSums, outputWeightSums, currentErrors);
        }

        if (this.currentEpoch % 10 === 0) {
          console.log(`Epoch: ${this.currentEpoch}`);
          console.log("o: ", this._expectedOutput, " a: ", actualOutput, " errors: ", e);
        }
      }
    }
    this.training = false;
    this.trained = true;
  }

  public updateInputWeights(epoch: number = 0,
                            x: number[],
                            hiddenWeightSum: number[],
                            outputWeightSum: number[],
                            errors: number[]) {

    const gradient: Matrix = this.getInputWeightsGradient(epoch, x, hiddenWeightSum, outputWeightSum, errors);
    const weights: Matrix = this.inputWeights;
    const learnRate: number = this.getLearnRate();
    // console.log("gradient: \n", gradient.toString());
    this.updateWeights(gradient, weights, learnRate);
  }

  private getInputWeightsGradient(epoch: number,
                                  x: number[],
                                  hiddenWeightSum: number[],
                                  outputWeightSum: number[],
                                  errors: number[]): Matrix {

    const gradientRowsCount = this.configuration.inputCount + this.configuration.contextCount;
    const gradientColumnsCount = this.configuration.hiddenCount;

    const gradient = new Matrix(gradientRowsCount, gradientColumnsCount);

    const w1: Matrix = this.inputWeights;

    let str = "";

    str += `epoch: ${epoch}\n` +
      `x: ${x}\n` +
      `hiddenWeightSum: ${hiddenWeightSum}\n`
      + `outputWeightSum: ${outputWeightSum}\n`
      + `errors: ${errors}\n`
      + `weights: ${w1}\n`;

    // delta 1
    for (let row = 0; row < gradientRowsCount; row++) {
      for (let column = 0; column < gradientColumnsCount; column++) {

        str += `${row}, ${column}\n`;

        //calculate gradient
        let gradientValue = 0;
        for (let s = 0; s < this.configuration.outputCount; s++) {
          str += `\ts = ${s}\n`;
          let e_s = errors[s];
          str += `\tes = ${e_s}\n`;
          let d_g_s = outputWeightSum[s];
          str += `\tdgs = ${d_g_s}\n`;
          let sum = 0;
          str += `\tsum = ${sum}\n`;
          for (let i = 0; i < this.configuration.hiddenCount; i++) {
            str += `\t\ti =  ${i}\n`;
            let u_i = hiddenWeightSum[i];
            str += `\t\tui =  ${u_i}\n`;
            let d_u_i = MathUtils.dSigmoid(u_i); //fixme
            str += `\t\tdui =  ${d_u_i}\n`;
            let dab_xb = MathUtils.kroneckerDelta(i, column) * x[row];
            str += `\t\tdab_xb =  ${dab_xb}\n`;
            if (epoch > 0) { // dv_dw(0) = 0
              for (let k = 0; k < this.configuration.hiddenCount; k++) {
                // console.log("index: ", k + this.configuration.inputCount);
                // console.log("weights: ", w1);
                // console.log("row: ", w1.getRow(k + this.configuration.inputCount))
                // console.log("item: ", w1.get(k + this.configuration.inputCount, i));
                dab_xb += MathUtils.dSigmoid(hiddenWeightSum[k]) * w1.get(k + this.configuration.inputCount, i);
                str += `\t\t\tk = ${k}, dab_xb = ${dab_xb}\n`;
              }
            }
            dab_xb = Math.floor(dab_xb);
            str += `\t\t dab_xb = ${dab_xb}\n`;
            sum += d_u_i * dab_xb;
            str += `\t\t sum = ${sum}\n`;
          }
          gradientValue += e_s * d_g_s * sum;
          str += `\tgradient ab: ${gradientValue} \n`;
        }
        gradient.set(row, column, gradientValue);
        str += `gradient [a][b]: ${gradient.get(row, column)}\n`;
      }
    }

    str += `gradient: ${gradient}\n"============================="\n\n`;

    // console.log(str);

    return gradient;
  }

  public updateHiddenWeights(hiddenWeightSum: number[],
                             outputWeightSum: number[],
                             errors: number[]) {

    const gradient: Matrix = this.getHiddenWeightsGradient(hiddenWeightSum, outputWeightSum, errors);
    const weights: Matrix = this.hiddenWeights;
    const learnRate: number = this.getLearnRate();
    this.updateWeights(gradient, weights, learnRate);
  }

  private getHiddenWeightsGradient(hiddenWeightSum: number[],
                                   outputWeightSum: number[],
                                   errors: number[]): Matrix {

    const gradientRowsCount = this.configuration.hiddenCount;
    const gradientColumnsCount = this.configuration.outputCount;

    const gradient = new Matrix(gradientRowsCount, gradientColumnsCount);

    // delta 1
    for (let row = 0; row < gradientRowsCount; row++) {
      for (let column = 0; column < gradientColumnsCount; column++) {

        //calculate gradient
        let gradientValue = errors[column]
          * MathUtils.dSigmoid(outputWeightSum[column])
          * MathUtils.sigmoid(hiddenWeightSum[row]);

        gradient.set(row, column, gradientValue);
      }
    }

    return gradient;
  }

  private updateWeights(gradient: Matrix, weights: Matrix, learnRate: number) {
    const {rowsCount, columnsCount} = weights.size();
    for (let i = 0; i < rowsCount; i++) {
      for (let j = 0; j < columnsCount; j++) {
        let newWeight = weights.get(i, j) - learnRate * gradient.get(i, j);
        weights.set(i, j, newWeight);
      }
    }
  }

  public feedWardPropagation() {

  }

  public backWardPropagation() {

  }

  private getLearnRate(): number {
    return 0.01;
  }

  private distribute(samples: Array<number[]>, yTrue: number[]): void {
    if (samples && samples.length > 0 && yTrue && yTrue.length > 0) {
      let length = samples.length;
      let trainingLength = this.getNewLength(length, this.configuration.training);
      let validationLength = this.getNewLength(length, this.configuration.validation);
      //samples
      this._trainingSamples = samples.slice(0, trainingLength);
      this._validationSamples = samples.slice(trainingLength, trainingLength + validationLength);
      this._testingSamples = samples.slice(trainingLength + validationLength, length);
      //output
      this._expectedOutput = yTrue.slice(0, trainingLength).map(y => [y]);

      // console.log(this._trainingSamples);
      // console.log(this._validationSamples);
      // console.log(this._testingSamples);
      // console.log(this._expectedOutput);
    }
  }

  private getNewLength(length: number = 0, percent: number = 0): number {
    if (length !== 0 && percent !== 0) {
      return Math.floor((percent / 100) * length);
    }
    return 0;
  }

  public normalizeVector(vector: number[]): number[] {
    let newVector = [];
    let min = vector[0];
    let max = vector[0];
    for (let iter = 0; iter < 2; iter++) {
      for (let i = 0; i < vector.length; i++) {
        let element = vector[i];
        if (iter === 0) {
          if (element > max) max = element;
          if (element < min) min = element;
        } else if (iter === 1) {
          let minimax = (element - min) / (max - min);
          newVector.push(minimax)
        }
      }
    }
    return newVector;
  }

  public normalizeMatrix(matrix: Array<number[]>): Array<number[]> {
    let newMatrix = [];
    let min = matrix[0][0];
    let max = matrix[0][0];
    for (let iter = 0; iter < 2; iter++) {
      for (let i = 0; i < matrix.length; i++) {
        let newRow = [];
        for (let j = 0; j < matrix[0].length; j++) {
          let element = matrix[i][j];
          if (iter === 0) {
            if (element > max) max = element;
            if (element < min) min = element;
          } else if (iter === 1) {
            let minimax = (element - min) / (max - min);
            newRow.push(minimax);
          }
        }
        if (iter === 1) {
          newMatrix.push(newRow);
        }
      }
    }
    return newMatrix;
  }

  public datasetToVector(): void {
    this.loading = true;
    this.loadDataset().subscribe((dataset) => {
      let lines = dataset.split("\n")
        .map(line => line.trim())
        .filter(line => line !== "");
      let classes = new Set();
      let subjectParamsVector: Array<number[]> = [];
      let expectedOutputVector = [];
      for (const lineStr of lines) {
        let line = lineStr.split(" ").map(value => value.trim());
        line = line.slice(1, line.length);
        let clazz = line[line.length - 1];
        classes.add(clazz);
        let lineVector: number[] = line.slice(0, line.length - 1)
          .map(element => Number.parseFloat(element));
        subjectParamsVector.push(lineVector);
        expectedOutputVector.push(this.indexOf(clazz, classes));
      }
      subjectParamsVector = this.normalizeMatrix(subjectParamsVector);
      expectedOutputVector = this.normalizeVector(expectedOutputVector);
      this.configuration.outputCount = 1;
      this.configuration.inputCount = subjectParamsVector[0].length;
      this.configuration.hiddenCount = Math.round((this.configuration.inputCount + this.configuration.outputCount) / 2);
      this.configuration.contextCount = this.configuration.hiddenCount;
      this.distribute(subjectParamsVector, expectedOutputVector);
      this.loading = false;
    }, (error) => {
      console.error(error);
    })
  }

  private indexOf(value: any, set: Set<any>): number {
    let i = 0;
    for (const element of set) {
      if (element === value) {
        return i;
      }
      i++;
    }
    return -1;
  }

  private loadDataset(): Observable<any> {
    return this.http.get("../../assets/irisDataset.txt", {
      responseType: "text"
    });
  }

}
