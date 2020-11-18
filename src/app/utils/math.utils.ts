export class MathUtils {
  public static sigmoid(x: number): number {
    return 1 / (1 - Math.exp(-x));
  }

  public static dSigmoid(x: number): number {
    let fx = this.sigmoid(x);
    return fx * (1 - fx);
  }

  public static tanh(x: number): number {
    return Math.tanh(x);
  }

  public static dTanh(x: number): number {
    return 1 / Math.pow(Math.cosh(x), 2);
  }

  public static mseLoss(yTrue: number[], yPred: number[]): number {
    let count = yTrue.length;
    let sum = 0;
    for (let i = 0; i < count; i++) {
      let sub = yTrue[i] - yPred[i];
      sum += Math.pow(sub, 2);
    }
    return sum / 2;
  }

  public static kroneckerDelta(i: number, j: number): number {
    return i == j ? 1 : 0;
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

  size(): { rowsCount: number, columnsCount: number } {
    let size = {rowsCount: 0, columnsCount: 0};
    if (this._rows.length === 0) {
      return size;
    }
    size.rowsCount = this._rows.length;
    size.columnsCount = this._rows[0].length;
    return size;
  }

}
