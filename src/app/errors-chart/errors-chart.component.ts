import {Component, Input, OnInit} from '@angular/core';
import {ChartDataSets, ChartOptions} from 'chart.js';
import {Color, Label} from 'ng2-charts';

@Component({
  selector: 'errors-chart',
  templateUrl: './errors-chart.component.html',
  styleUrls: ['./errors-chart.component.less']
})
export class ErrorsChartComponent implements OnInit {

  @Input()
  public lineChartData: ChartDataSets[] = [{
    data: [],
    label: 'Error',
    borderWidth: 1
  }];
  @Input()
  public lineChartLabels: Label[] = [];

  public lineChartOptions = {
    responsive: true,
    elements: {
      point: {
        borderWidth: 0
      }
    }
  };
  public lineChartColors: Color[] = [{
    borderColor: 'black',
    backgroundColor: 'rgba(255,0,0,0.3)',
  }];
  public lineChartLegend = true;
  public lineChartType = 'line';
  public lineChartPlugins = [];

  constructor() {
  }

  ngOnInit(): void {
  }

}
