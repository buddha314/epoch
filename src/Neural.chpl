/*
 Another pass at the NN based on Alg. 6.3, 6.4 in Goodfellow, et. al. Chapter 6, approx p 204
 */

 module Neural {
   use LinearAlgebra,
       Time,
       Norm,
  //     Model,
       Random;

   class Layer {
     var wDom: domain(2),
         bDom: domain(1),
         W:[wDom] real,
         b:[bDom] real,
         g: Activation;


     proc init(activation: string, udim: int, ldim: int){
       this.wDom = {1..udim,1..ldim};
       this.bDom = {1..udim};
       var W: [wDom] real;
       var b: [bDom] real;
       this.W = W;
       this.b = b;
       fillRandom(this.W);
       fillRandom(this.b);
       this.W = 0.1*this.W;
       this.b = 0.1*this.b;
       this.g = new Activation(name=activation);
     }
/*
     proc readWriteThis(f) throws {
       f <~> "%6s".format(this.name)
         <~> " W:" <~> this.W.shape
         <~> " h:" <~> this.h.shape
         <~> " b:" <~> this.b.shape
         <~> " a:" <~> this.a.shape;
*/

     proc linearForward(A_prev: []) {
       const zDom: domain(2) = {this.W.domain.dim(1),A_prev.domain.dim(2)};
       var b: [zDom] real;
       for i in zDom.dim(2) {
         b[..,i] = this.b;
       }
       const Z = (this.W.dot(A_prev)).plus(b);
       return Z;
     }

   }

   class Activation {
     var name: string;
     proc init(name: string) {
       this.name=name;
     }

     proc f(x: real) {
       if this.name == "relu" {
         return ramp(x);
       } else if this.name == "sigmoid" {
         return sigmoid(x);
       } else if this.name == "tanh" {
        return tanh(x);
       } else if this.name == "step" {
         return heaviside(x);
       } else {
         return 0;
       }
     }

     proc df(x:real) {
       if this.name == "relu" {
         return dramp(x);
       } else if this.name == "sigmoid" {
         return dsigmoid(x);
       } else if this.name == "tanh" {
         return dtanh(x);
       } else if this.name == "step" {
         return dsigmoid(x);  //maybe I'll make this dsigmoid(x) for fun?
       } else {
         return 0;
       }
     }

     // Activation Functions
     proc ramp(x: real) {
       if x < 0 {
         return 0;
       } else {
         return x;
       }
     }

     proc sigmoid(x: real) {
       return (1/(1 + exp(-x)));
     }

     proc tanh(x: real) {
       return (exp(x) - exp(-x))/(exp(x) + exp(-x));
     }

     proc heaviside(x) {
       if x < 0 {
         return 0;
       } else {
         return 1;
       }
     }

     // Derivates of Activation Functions
     proc dsigmoid(x) {
       return sigmoid(x) * (1 - sigmoid(x));
     }

     proc dramp(x) {
       return heaviside(x);
     }

     proc dtanh(x) {
       return 1 - (tanh(x))**2;
     }

     proc dheaviside(x) {
       if x == 0 {
         return 10000000000000000;
       } else {
         return 0;
       }
     }

  }

  class Dense {
    var units: int,
        inputDim: int;

    proc init(units:int, inputDim=0) {
      this.units=units;
      this.inputDim=inputDim;
    }
  }

  class Loss {
    var name: string;
    proc init(name: string="DEFAULT") {
      this.name = name;
    }
    proc J(yHat: [], y:[]) {
      var r: [yHat.domain] real;
      if this.name == "DEFAULT" {
        r = yHat - y;
      } else {
        r = yHat - y;
      }
      return r;
    }
  }

}
