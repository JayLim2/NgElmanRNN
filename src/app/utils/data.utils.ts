import {HttpClient} from "@angular/common/http";
import {Injectable} from "@angular/core";

@Injectable()
export class DataUtils {

  private _currentDataset: string = null;

  constructor(
    private http: HttpClient
  ) {
  }

  get currentDataset(): string {
    return this._currentDataset;
  }

  public datasetToVector() {
    this.http.get("../../assets/irisDataset.txt", {
      responseType: "text"
    }).subscribe((dataset) => {
      this._currentDataset = dataset;
    }, (error) => {
      console.error(error);
    })
  }

}
