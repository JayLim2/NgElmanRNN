import {Component, OnInit} from '@angular/core';
import {DataUtils} from "./utils/data.utils";

/*
    Пусть скрытых слоев = 1,
          входных слоев = 1,
          выходных слоев = 1

    Пусть на входе - 4 нейрона (= 4 признака)
          на выходе - 1 нейрон (= 1 ответ)
          на скрытом - (4+1)/2 = ~3 нейрона

    Тогда
    матрица весов на входе = вектор длиной 3
      [[w11, w12, w13], [w21, w22, w23],
       [w31, w32, w33], [w41, w42, w43]]
    матрица весов на скрытом = вектор длиной 3
      [[hw1], [hw2], [hw3]]
*/

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.less']
})
export class AppComponent implements OnInit {
  title = 'NgElmanRNN 0.1-snapshot';

  constructor(
    public dataUtils: DataUtils
  ) {
  }

  ngOnInit() {
    this.dataUtils.datasetToVector();
  }

  feedWardPropagation() {

  }

  backWardPropagation() {

  }

}
