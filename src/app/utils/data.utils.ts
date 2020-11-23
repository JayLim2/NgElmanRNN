import {HttpClient} from "@angular/common/http";
import {Injectable} from "@angular/core";
import {Observable, Subject} from "rxjs";
import {MathDecorator, MathUtils, Matrix} from "./math.utils";

@Injectable()
export class DataUtils {

  private _trainingSamples: Array<number[]>;
  private _testingSamples: Array<number[]>;

  private _expectedOutput: Array<number[]>;

  private configuration = {
    epochs: 700,
    training: 90,
    testing: 0,
    inputCount: 0,
    contextCount: 0,
    hiddenCount: 0,
    outputCount: 0,
    normalize: false,
    moment: false
  };

  output: string = "";
  outputS: Subject<string> = new Subject<string>();

  private _hiddenWeights: Matrix = null;
  private _outputWeights: Matrix = null;

  public loading: boolean = false;

  public training: boolean = false;
  public trained: boolean = false;

  public lineChartData = [{
    data: [],
    label: 'Error'
  }];
  public lineChartLabels = [];

  private lastDeltaW1: Matrix = null;
  private lastDeltaW2: Matrix = null;

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

  random(left: number, right: number): number {
    return Math.random() * (right - left) + left;
  }

  get hiddenWeights(): Matrix {
    if (!this._hiddenWeights) {
      let matrix = new Matrix(
        this.configuration.inputCount + this.configuration.contextCount,
        this.configuration.hiddenCount
      );
      for (let i = 0; i < this.configuration.inputCount + this.configuration.hiddenCount; i++) {
        for (let j = 0; j < this.configuration.hiddenCount; j++) {
          matrix.set(i, j, this.random(0, 1));
        }
      }
      this._hiddenWeights = matrix;
    }
    return this._hiddenWeights;
  }

  get outputWeights(): Matrix {
    if (!this._outputWeights) {
      let matrix = new Matrix(
        this.configuration.hiddenCount,
        this.configuration.outputCount
      );
      for (let i = 0; i < this.configuration.hiddenCount; i++) {
        for (let j = 0; j < this.configuration.outputCount; j++) {
          matrix.set(i, j, Math.random());
        }
      }
      this._outputWeights = matrix;
    }
    return this._outputWeights;
  }

  public configure(epochs: number = 1000, training: number = 70) {
    if (epochs <= 0) {
      throw new Error("Epochs count must be greater than 0.");
    }
    if (training <= 0) {
      throw new Error("Samples size must be greater than 0.");
    }
    if (training > 99) {
      throw new Error("Training and testing percents must summary equals 100.");
    }
    this.configuration.training = training;
    this.configuration.testing = 100 - training;
  }

  /**
   * Основной метод: запускает обучения, а затем тестирование и выводит результаты
   */
  public test() {
    let testing = this.testingByClass;
    for (const className of Object.keys(testing)) {
      let byClass = testing[className];
      for (let i = 0; i < byClass.length; i++) {
        let testingSample = byClass[i];
        this.output += `${className}\ninput: ${testingSample}\n`;
        this.outputS.next(this.output);
        this.feedWardPropagation(testingSample, className);
      }
    }
  }

  public currentEpoch;

  public currentEpochSubject = new Subject();

  public currentEpoch$ = this.currentEpochSubject.asObservable();

  public train(moment: boolean = false) {
    this.configuration.moment = moment;
    if (this.configuration.moment) {
      this.lastDeltaW1 = null;
      this.lastDeltaW2 = null;
    }

    this._hiddenWeights = null;
    this._outputWeights = null;
    this.output = "";
    this.outputS.next(this.output);
    this.currentEpoch = 0;
    this.lineChartData[0].data = [];
    this.lineChartLabels = [];

    this.training = true;
    const epochs = this.configuration.epochs;
    if (epochs > 0) {
      const inputCount = this.configuration.inputCount;
      const contextCount = this.configuration.contextCount;

      //iterate by epochs
      for (this.currentEpoch = 0; this.currentEpoch < epochs; this.currentEpoch++) {
        this.currentEpochSubject.next(this.currentEpoch);
        let x: number[] = Array(inputCount + contextCount).fill(0);
        //iterate by training samples
        let actualOutput = [];
        let e = [];
        for (let index = 0; index < this.trainingSamples.length; index++) {
          if(this.hiddenWeights.containsNaN() || this.outputWeights.containsNaN()) {
            console.error("NaN detected.");
            return;
          }
          let currentSample: number[] = this.trainingSamples[index];
          for (let i = 0; i < inputCount; i++) {
            x[i] = currentSample[i];
          }
          let w: Matrix = this.hiddenWeights;

          //iterate by hidden neurons
          let hiddenWeightSums = [];
          for (let i = 0; i < this.configuration.hiddenCount; i++) {
            let sum = 0;
            for (let j = 0; j < this.configuration.inputCount + this.configuration.contextCount; j++) {
              sum += w.get(j, i) * x[j];
            }
            hiddenWeightSums[i] = sum;
            x[inputCount + i] = MathDecorator.function1(hiddenWeightSums[i]);
          }

          //iterate by output neurons
          w = this.outputWeights;
          let outputWeightSums = [];
          let sampleOutput = [];
          for (let s = 0; s < this.configuration.outputCount; s++) {
            let sum = 0;
            for (let i = 0; i < this.configuration.hiddenCount; i++) {
              sum += w.get(i, s) * x[inputCount + i];
            }
            outputWeightSums[s] = sum;
            sampleOutput[s] = MathDecorator.function2(outputWeightSums[s]);
          }
          actualOutput = [...actualOutput, sampleOutput];

          //error
          let currentErrors = [];
          let resultError = 0;
          for (let i = 0; i < this.configuration.outputCount; i++) {
            let error = this._expectedOutput[index][i] - sampleOutput[i];
            currentErrors[i] = error;
            resultError += Math.pow(error, 2);
          }

          if (resultError) {
            e.push(resultError);
          }

          // корректировка весов
          this.updateHiddenWeights(this.currentEpoch, x, hiddenWeightSums, outputWeightSums, currentErrors);
          this.updateOutputWeights(hiddenWeightSums, outputWeightSums, currentErrors);
        }

        this.lineChartData[0].data.push(MathUtils.standardDeviation(e));
        this.lineChartLabels.push(`${this.currentEpoch}`);

        if (this.currentEpoch % 50 === 0) {
          console.log(`Epoch: ${this.currentEpoch}`);
          console.log("o: ", this._expectedOutput, " a: ", actualOutput, " errors: ", e);
        }
      }
    }
    this.training = false;
    this.trained = true;
  }

  public updateHiddenWeights(epoch: number = 0,
                             x: number[],
                             hiddenWeightSum: number[],
                             outputWeightSum: number[],
                             errors: number[]) {

    const gradient: Matrix = this.getHiddenWeightsGradient(epoch, x, hiddenWeightSum, outputWeightSum, errors);
    const weights: Matrix = this.hiddenWeights;
    const learnRate: number = this.getLearnRate();
    if (!this.lastDeltaW1) {
      this.lastDeltaW1 = new Matrix(weights.size().rowsCount, weights.size().columnsCount);
      for (let i = 0; i < this.lastDeltaW1.size().rowsCount; i++) {
        for (let j = 0; j < this.lastDeltaW1.size().columnsCount; j++) {
          this.lastDeltaW1.set(i, j, 0);
        }
      }
    }
    this.updateWeights("input", gradient, weights, learnRate);
  }

  private getHiddenWeightsGradient(epoch: number,
                                  x: number[],
                                  hiddenWeightSum: number[],
                                  outputWeightSum: number[],
                                  errors: number[]): Matrix {

    const gradientRowsCount = this.configuration.inputCount + this.configuration.contextCount;
    const gradientColumnsCount = this.configuration.hiddenCount;

    const gradient = new Matrix(gradientRowsCount, gradientColumnsCount);

    const w1: Matrix = this.hiddenWeights;
    const w2: Matrix = this.outputWeights;

    // delta 1
    for (let row = 0; row < gradientRowsCount; row++) {
      for (let column = 0; column < gradientColumnsCount; column++) {

        //calculate gradient
        let gradientValue = 0;
        for (let s = 0; s < this.configuration.outputCount; s++) {
          let e_s = errors[s];
          let g_s = outputWeightSum[s];
          let dfg_dg = MathDecorator.derivative2(g_s);
          let sum = 0;
          if (epoch > 0) { // dv_dw(0) = 0
            for (let i = 0; i < this.configuration.hiddenCount; i++) {
              for (let k = 0; k < this.configuration.hiddenCount; k++) {
                let u_i = hiddenWeightSum[i];
                let d_u_i = MathDecorator.derivative1(u_i);
                let dab_xb = MathUtils.kroneckerDelta(i, column) * x[row];
                dab_xb += MathDecorator.derivative2(hiddenWeightSum[k]) * w1.get(k + this.configuration.inputCount, i);
                sum += d_u_i * dab_xb * w2.get(i, s);
              }
            }
          }
          gradientValue += e_s * dfg_dg * sum;
        }
        gradient.set(row, column, gradientValue);
      }
    }

    return gradient;
  }

  public updateOutputWeights(hiddenWeightSum: number[],
                             outputWeightSum: number[],
                             errors: number[]) {

    const gradient: Matrix = this.getOutputWeightsGradient(hiddenWeightSum, outputWeightSum, errors);
    const weights: Matrix = this.outputWeights;
    const learnRate: number = this.getLearnRate();
    if (!this.lastDeltaW2) {
      this.lastDeltaW2 = new Matrix(weights.size().rowsCount, weights.size().columnsCount);
      for (let i = 0; i < this.lastDeltaW2.size().rowsCount; i++) {
        for (let j = 0; j < this.lastDeltaW2.size().columnsCount; j++) {
          this.lastDeltaW2.set(i, j, 0);
        }
      }
    }
    this.updateWeights("hidden", gradient, weights, learnRate);
  }

  private getOutputWeightsGradient(hiddenWeightSum: number[],
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
          * MathDecorator.derivative2(outputWeightSum[column])
          * MathDecorator.function2(hiddenWeightSum[row]);

        gradient.set(row, column, gradientValue);
      }
    }

    return gradient;
  }

  private alpha(): number {
    return 0.01;
  }

  private updateWeights(name: string, gradient: Matrix, weights: Matrix, learnRate: number) {
    const {rowsCount, columnsCount} = weights.size();
    for (let i = 0; i < rowsCount; i++) {
      for (let j = 0; j < columnsCount; j++) {
        let oldWeight = weights.get(i, j);
        let newWeight = oldWeight + learnRate * gradient.get(i, j);
        if (this.configuration.moment) {
          if (name === "input") {
            newWeight += this.alpha() * this.lastDeltaW1.get(i, j);
            this.lastDeltaW1.set(i, j, newWeight - oldWeight);
          } else if(name === "hidden") {
            newWeight += this.alpha() * this.lastDeltaW2.get(i, j);
            this.lastDeltaW2.set(i, j, newWeight - oldWeight);
          }
        }
        weights.set(i, j, newWeight);
      }
    }
  }

  public feedWardPropagation(x: number[], name?: string) {
    const inputCount = x.length;

    x = [...x, ...Array(this.configuration.contextCount).fill(0)];

    //iterate by hidden neurons
    let w: Matrix = this.hiddenWeights;
    for (let i = 0; i < this.configuration.hiddenCount; i++) {
      let sum = 0;
      for (let j = 0; j < this.configuration.inputCount + this.configuration.contextCount; j++) {
        sum += w.get(j, i) * x[j];
      }
      x[inputCount + i] = MathDecorator.function1(sum);
    }

    //iterate by output neurons
    w = this.outputWeights;
    let currentOutput = [];
    for (let s = 0; s < this.configuration.outputCount; s++) {
      let sum = 0;
      for (let i = 0; i < this.configuration.hiddenCount; i++) {
        sum += w.get(i, s) * x[inputCount + i];
      }
      currentOutput[s] = MathDecorator.function2(sum);
    }

    let index = 0;
    let max = currentOutput[index];
    for (let i = 0; i < currentOutput.length; i++) {
      if (currentOutput[i] > max) {
        index = i;
        max = currentOutput[index];
      }
    }

    let isTrue = false;
    if (name === "setosa") {
      isTrue = index === 0;
    } else if(name === "versicolor") {
      isTrue = index === 1;
    } else if(name === "virginica") {
      isTrue = index === 2;
    }

    this.output += `output: ${currentOutput} [${isTrue}]\n\n`;
    this.outputS.next(this.output);
  }

  private getLearnRate(): number {
    return 0.1;
  }

  testingByClass = {
    "setosa": [],
    "versicolor": [],
    "virginica": []
  }

  private distribute(samples: Array<number[]>, yTrue: number[]): void {
    if (samples && samples.length > 0 && yTrue && yTrue.length > 0) {
      const classes = [
        50, //setosa
        49, //versicolor,
        51, //virginica
      ];

      let trainingSamples = [];
      let expectedOutput = [];

      let handled = 0;
      for (let i = 0; i < classes.length; i++) {
        let count = classes[i];
        let samplesByClass = samples.slice(handled, handled + count);
        let yTrueByClass = yTrue.slice(handled, handled + count);
        handled += count;

        let length = samplesByClass.length;
        let trainingLength = this.getNewLength(length, this.configuration.training);
        //samples
        let trainingSamplesByClass = samplesByClass.slice(0, trainingLength);
        trainingSamples = [
          ...trainingSamples,
          ...trainingSamplesByClass
        ];
        let testingSamplesByClass = samplesByClass.slice(trainingLength, length);
        let name = "unknown";
        switch (i) {
          case 0:
            name = "setosa";
            break;
          case 1:
            name = "versicolor";
            break;
          case 2:
            name = "virginica";
            break;
        }
        this.testingByClass[name] = [...testingSamplesByClass];
        //output
        for (let j = 0; j < trainingSamplesByClass.length; j++) {
          let output = Array(this.configuration.outputCount).fill(0);
          output[i] = 1;
          expectedOutput.push(output);
        }
      }

      this._trainingSamples = trainingSamples;
      this._expectedOutput = expectedOutput;
    }
  }

  private getNewLength(length: number = 0, percent: number = 0): number {
    if (length !== 0 && percent !== 0) {
      return Math.floor((percent / 100) * length);
    }
    return 0;
  }

  public normalizeVector(vector: number[]): number[] {
    if (this.configuration.normalize) {
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
    return vector;
  }

  public normalizeMatrix(matrix: Array<number[]>): Array<number[]> {
    if (this.configuration.normalize) {
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
    return matrix;
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
      this.configuration.outputCount = 3;
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
