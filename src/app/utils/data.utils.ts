import {HttpClient} from "@angular/common/http";
import {Injectable} from "@angular/core";
import {Observable} from "rxjs";

@Injectable()
export class DataUtils {

  private _currentDataset: string = null;

  private _trainingSamples: Array<number[]>;
  private _validationSamples: Array<number[]>;
  private _testingSamples: Array<number[]>;

  private _expectedOutput: number[];

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

  constructor(
    private http: HttpClient
  ) {
  }

  get currentDataset(): string {
    return this._currentDataset;
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
      for (let i = 0; i < this.configuration.inputCount; i++) {
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

  public train() {
    this.datasetToVector();
    const learnRate = 0.1;
    const epochs = this.configuration.epochs;
    if (epochs > 0) {
      const inputCount = this.configuration.inputCount;
      const contextCount = this.configuration.contextCount;

      //iterate by epochs
      for (let currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {
        let x: number[] = Array(inputCount + contextCount).fill(0);
        // console.log(inputCount, " ", contextCount);
        // console.log("X = ", x);
        //iterate by training samples
        let actualOutput = [];
        for (let index = 0; index < this.trainingSamples.length; index++) {
          let currentSample: number[] = this.trainingSamples[index]; // vector, [x1, x2, x3, x4]
          for (let i = 0; i < inputCount; i++) {
            x[i] = currentSample[i];
          }
          let w: Matrix = this.inputWeights;

          // console.log(w);

          //iterate by hidden neurons
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
            x[inputCount + i] = this.sigmoid(sum);
          }

          //iterate by output neurons
          w = this.hiddenWeights;
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
            currentOutput[s] = this.sigmoid(sum);
          }
          actualOutput = [...actualOutput, ...currentOutput]; //fixme

          let error = this.mseLoss(this._expectedOutput, actualOutput);

          if (currentEpoch % 10 === 0) {
            console.log(`Epoch: ${currentEpoch}`);
            console.log("o: ", this._expectedOutput, " ", actualOutput, " ", error);
          }
        }
      }
    }
  }

  public feedWardPropagation() {

  }

  public backWardPropagation() {

  }

  private sigmoid(x: number): number {
    return 1 / (1 - Math.exp(-x));
  }

  private dSigmoid(x: number): number {
    let fx = this.sigmoid(x);
    return fx * (1 - fx);
  }

  private mseLoss(yTrue: number[], yPred: number[]): number {
    // console.log("y: ", yTrue, " ", yPred);
    let count = yTrue.length;
    let sum = 0;
    for (let i = 0; i < count; i++) {
      let sub = yTrue[i] - yPred[i];
      sum += Math.pow(sub, 2);
    }
    return sum / 2;
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
      this._expectedOutput = yTrue.slice(0, trainingLength);

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

  public loading: boolean = false;

  public datasetToVector(): void {
    this.loading = true;
    this.loadDataset().subscribe((dataset) => {
      this._currentDataset = dataset;

      let lines = dataset.split("\n")
        .map(line => line.trim())
        .filter(line => line !== "");
      let classes = new Set();
      let subjectParamsVector = [];
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

export class Matrix {

  private _rows: Array<number[]> = [];

  constructor(rows: number = 0, columns?: number) {
    if (rows > 0) {
      if (!columns || columns <= 0) {
        columns = rows;
      }
      for (let i = 0; i < rows; i++) {
        this._rows[i] = Array<number>(columns).fill(0);
      }
    }
  }

  get rowsCount(): number {
    return this._rows.length;
  }

  get columnsCount(): number {
    return this._rows[0].length;
  }

  set(row: number, column: number, value: number): void {
    this._rows[row][column] = value;
  }

  getRow(row: number): number[] {
    return this._rows[row];
  }

  get(row: number, column: number): number {
    return this._rows[row][column];
  }

}
