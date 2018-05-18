








module Model {
  use Neural;

  class Model {

    proc init() {}

    proc init(modelType: string = "fcnetwork", dims: [] int, activations: [] string) {
      var model = new FCNetwork(dims, activations);
    }

    proc predict(X:[]) {
      if this: FCNetwork then
        return (this:FCNetwork).forwardPass(X);
      else
        halt("I don't know this model type");
    }

    proc inputDim() {
      if this: FCNetwork then
        return (this:FCNetwork).widths[widths.domain.first];
      else
        halt("I don't know this model type");
    }

    proc outputDim() {
      if this: FCNetwork then
        return (this:FCNetwork).widths[widths.domain.last];
      else
        halt("I don't know this model type");
    }
  }

/*
  class Dojo {
    var epochs: int,
        reportInterval: int,
        momentum: real,
        regularizer: string;

    proc Model.train() { }

  }
*/

}
