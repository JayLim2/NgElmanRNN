import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ErrorsChartComponent } from './errors-chart.component';

describe('ErrorsChartComponent', () => {
  let component: ErrorsChartComponent;
  let fixture: ComponentFixture<ErrorsChartComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ErrorsChartComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ErrorsChartComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
