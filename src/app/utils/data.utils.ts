import {HttpClient} from "@angular/common/http";
import {Injectable} from "@angular/core";

@Injectable()
export class DataUtils {

  private _currentDataset: string = null;

  private _trainingSamples: number[];
  private _validationSamples: number[];
  private _testingSamples: number[];

  private configuration = {
    training: 70,
    validation: 15,
    testing: 15,
    inputCount: 0,
    contextCount: 0,
    hiddenCount: 0,
    outputCount: 0
  };

  constructor(
    private http: HttpClient
  ) {
  }

  get currentDataset(): string {
    return this._currentDataset;
  }

  get trainingSamples(): number[] {
    return this._trainingSamples;
  }

  get validationSamples(): number[] {
    return this._validationSamples;
  }

  get testingSamples(): number[] {
    return this._testingSamples;
  }

  public configure(training: number = 70, validation: number = 15) {
    if (training + validation > 100) {
      throw new Error("Training, testing and validation percents must summary equals 100.");
    }
    this.configuration.training = training;
    this.configuration.validation = validation;
    this.configuration.testing = 100 - training - validation;
  }

  public distribute(samples: number[]) {
    if (samples && samples.length > 0) {
      let length = samples.length;
      let trainingLength = this.getNewLength(length, this.configuration.training);
      let validationLength = this.getNewLength(length, this.configuration.validation);
      this._trainingSamples = samples.slice(0, trainingLength);
      this._validationSamples = samples.slice(trainingLength, trainingLength + validationLength);
      this._testingSamples = samples.slice(trainingLength + validationLength, length);

      console.log(this._trainingSamples);
      console.log(this._validationSamples);
      console.log(this._testingSamples);
    }
  }

  public getNewLength(length: number = 0, percent: number = 0) {
    if (length !== 0 && percent !== 0) {
      return Math.floor((percent / 100) * length);
    }
  }

  public datasetToVector() {
    this.getDataset().subscribe((dataset) => {
      this._currentDataset = dataset;

      let lines = dataset.split("\n")
        .map(line => line.trim())
        .filter(line => line !== "");
      let classes = new Set();
      let resultVector = [];
      for (const lineStr of lines) {
        let line = lineStr.split(" ").map(value => value.trim());
        line = line.slice(1, line.length);
        let clazz = line[line.length - 1];
        classes.add(clazz);
        let lineVector = [
          ...line.slice(0, line.length - 1).map(element => Number.parseFloat(element)),
          this.indexOf(clazz, classes)
        ];
        resultVector.push(lineVector);
      }
      this.configuration.outputCount = 1;
      this.configuration.inputCount = resultVector[0].length - this.configuration.outputCount;
      this.configuration.hiddenCount = Math.round((this.configuration.inputCount + this.configuration.outputCount) / 2);
      this.configuration.contextCount = this.configuration.hiddenCount;
      this.distribute(resultVector);

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

  private getDataset() {
    return this.http.get("../../assets/irisDataset.txt", {
      responseType: "text"
    });
  }

  private _inputWeights: Matrix = null;
  private _hiddenWeights: Matrix = null;

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

  get(row: number, column: number): number {
    return this._rows[row][column];
  }

}
