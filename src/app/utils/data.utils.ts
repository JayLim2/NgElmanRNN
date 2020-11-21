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
    epochs: 2000,
    training: 80,
    testing: 0,
    inputCount: 0,
    contextCount: 0,
    hiddenCount: 0,
    outputCount: 0,
    normalize: true,
    moment: true
  };

  output: string = "";
  outputS: Subject<string> = new Subject<string>();

  private _inputWeights: Matrix = null;
  private _hiddenWeights: Matrix = null;

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

  get testingSamples(): Array<number[]> {
    return this._testingSamples;
  }

  random(left: number, right: number): number {
    return Math.random() * (right - left) + left;
  }

  get inputWeights(): Matrix {
    if (!this._inputWeights) {
      let matrix = new Matrix(
        this.configuration.inputCount + this.configuration.contextCount,
        this.configuration.hiddenCount
      );
      for (let i = 0; i < this.configuration.inputCount + this.configuration.hiddenCount; i++) {
        for (let j = 0; j < this.configuration.hiddenCount; j++) {
          matrix.set(i, j, this.random(0, 1));
        }
      }
      // console.log(matrix.toString());
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
    let testing = [
      [5.2, 3.5, 1.5, 0.2], //setosa
      [4.8, 3.0, 1.4, 0.3], //setosa
      [6.7, 3.1, 4.4, 1.4], //versicolor
      [5.5, 2.5, 4.0, 1.3], //versicolor
      [7.2, 3.2, 6.0, 1.8], //virginica
      [6.5, 3.0, 5.5, 1.8], //virginica
    ];
    testing = this.normalizeMatrix(testing);
    testing = this.testingSamples;
    for (let i = 0; i < testing.length; i++) {
      let name = "";
      if (i >= 0 && i <= 10) {
        name = "setosa";
      } else if (i >= 11 && i < 20) {
        name = "versicolor";
      } else if (i >= 21) {
        name = "virginica";
      }
      let testingSample = testing[i];

      this.output += `${name}\ninput: ${testingSample}\n`;
      this.outputS.next(this.output);
      // console.log(name);
      // console.log("Input: ", testingSample);
      this.feedWardPropagation(testingSample);
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

    this._inputWeights = null;
    this._hiddenWeights = null;
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
          // console.log("x = ", x);
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
              sum += w.get(j, i) * x[j];
            }
            hiddenWeightSums[i] = sum;
            x[inputCount + i] = MathDecorator.function1(hiddenWeightSums[i]);
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
              sum += w.get(i, s) * x[inputCount + i];
            }
            outputWeightSums[s] = sum;
            currentOutput[s] = MathDecorator.function2(outputWeightSums[s]);
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

        // this.lineChartData[0].data.push(e[0]);
        this.lineChartData[0].data.push(MathUtils.standardDeviation(e));
        this.lineChartLabels.push(`${this.currentEpoch}`);

        if (this.currentEpoch % 50 === 0) {
          // console.log(`Epoch: ${this.currentEpoch}`);
          // console.log("o: ", this._expectedOutput, " a: ", actualOutput, " errors: ", e);
        }
      }
    }
    this.training = false;
    this.trained = true;

    console.log("w1: ", this.inputWeights.toString());
    console.log("w2: ", this.hiddenWeights.toString());
  }

  public updateInputWeights(epoch: number = 0,
                            x: number[],
                            hiddenWeightSum: number[],
                            outputWeightSum: number[],
                            errors: number[]) {

    // console.log(x);
    // console.log(errors);
    // console.log("===");


    const gradient: Matrix = this.getInputWeightsGradient(epoch, x, hiddenWeightSum, outputWeightSum, errors);
    const weights: Matrix = this.inputWeights;
    const learnRate: number = this.getLearnRate();
    // console.log("gradient: \n", gradient.toString());
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

  private getInputWeightsGradient(epoch: number,
                                  x: number[],
                                  hiddenWeightSum: number[],
                                  outputWeightSum: number[],
                                  errors: number[]): Matrix {

    const gradientRowsCount = this.configuration.inputCount + this.configuration.contextCount;
    const gradientColumnsCount = this.configuration.hiddenCount;

    const gradient = new Matrix(gradientRowsCount, gradientColumnsCount);

    const w1: Matrix = this.inputWeights;
    const w2: Matrix = this.hiddenWeights;

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
          let g_s = outputWeightSum[s];
          let dfg_dg = MathDecorator.derivative2(g_s);
          str += `\tdfg_dg = ${dfg_dg}\n`;
          let sum = 0;
          str += `\tsum = ${sum}\n`;
          if (epoch > 0) { // dv_dw(0) = 0
            for (let i = 0; i < this.configuration.hiddenCount; i++) {
              for (let k = 0; k < this.configuration.hiddenCount; k++) {
                str += `\t\ti =  ${i}\n`;
                let u_i = hiddenWeightSum[i];
                str += `\t\tui =  ${u_i}\n`;
                let d_u_i = MathDecorator.derivative1(u_i);
                str += `\t\tdui =  ${d_u_i}\n`;
                let dab_xb = MathUtils.kroneckerDelta(i, column) * x[row];
                str += `\t\tdab_xb =  ${dab_xb}\n`;

                // console.log("index: ", k + this.configuration.inputCount);
                // console.log("weights: ", w1);
                // console.log("row: ", w1.getRow(k + this.configuration.inputCount))
                // console.log("item: ", w1.get(k + this.configuration.inputCount, i));
                dab_xb += MathDecorator.derivative2(hiddenWeightSum[k]) * w1.get(k + this.configuration.inputCount, i);
                str += `\t\t\tk = ${k}, dab_xb = ${dab_xb}\n`;
                sum += d_u_i * dab_xb * w2.get(i, s);

              }
            }
            // dab_xb = Math.floor(dab_xb);
            // str += `\t\t dab_xb = ${dab_xb}\n`;
            str += `\t\t sum = ${sum}\n`;
          }
          gradientValue += e_s * dfg_dg * sum;
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
    // console.log("gradient H: \n", gradient.toString());
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
          * MathDecorator.derivative2(outputWeightSum[column])
          * MathDecorator.function2(hiddenWeightSum[row]);

        gradient.set(row, column, gradientValue);
      }
    }

    return gradient;
  }

  private alpha(): number {
    return 0.1;
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

  public feedWardPropagation(x: number[]) {
    const inputCount = x.length;

    x = [...x, ...Array(this.configuration.contextCount).fill(0)];

    //iterate by hidden neurons
    let w: Matrix = this.inputWeights;
    console.log("w1: ", w.toString());
    for (let i = 0; i < this.configuration.hiddenCount; i++) {
      let sum = 0;
      for (let j = 0; j < this.configuration.inputCount + this.configuration.contextCount; j++) {
        sum += w.get(j, i) * x[j];
      }
      x[inputCount + i] = MathDecorator.function1(sum);
    }

    //iterate by output neurons
    w = this.hiddenWeights;
    console.log("w2: ", w.toString());
    let currentOutput = [];
    for (let s = 0; s < this.configuration.outputCount; s++) {
      let sum = 0;
      for (let i = 0; i < this.configuration.hiddenCount; i++) {
        sum += w.get(i, s) * x[inputCount + i];
      }
      currentOutput[s] = MathDecorator.function2(sum);
    }

    this.output += `output: ${currentOutput}\n\n`;
    this.outputS.next(this.output);
    // console.log("Output: ", currentOutput);
  }

  public backWardPropagation() {

  }

  private getLearnRate(): number {
    return 0.2;
  }

  private distribute(samples: Array<number[]>, yTrue: number[]): void {
    if (samples && samples.length > 0 && yTrue && yTrue.length > 0) {
      const classes = [
        50, //setosa
        49, //versicolor,
        51, //virginica
      ];

      let trainingSamples = [];
      let testingSamples = [];
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
        trainingSamples = [
          ...trainingSamples,
          ...samplesByClass.slice(0, trainingLength)
        ];
        testingSamples = [
          ...testingSamples,
          ...samplesByClass.slice(trainingLength, length)
        ];
        // console.log("class: ", ...samplesByClass.slice(trainingLength, length));
        //output
        expectedOutput = [
          ...expectedOutput,
          ...yTrueByClass.slice(0, trainingLength).map(y => [y])
        ];
      }

      this._trainingSamples = trainingSamples;
      this._testingSamples = testingSamples;
      this._expectedOutput = expectedOutput;

      // console.log(this._trainingSamples);
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
