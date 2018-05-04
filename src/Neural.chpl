/*
 Another pass at the NN based on Alg. 6.3, 6.4 in Goodfellow, et. al. Chapter 6, approx p 204
 */

 module Neural {
   use LinearAlgebra,
       Time,
       Norm,
  //     Model,
       Random;


/*  A Fully Connected (FC) Neural Network is a stack of Layers  */
   class FCNetwork {
     var layerDom = {1..0},
         layers: [layerDom] Layer,
         caches: [layerDom] Cache, // Populated if `trained` = false
         dims: [layerDom] int,
         activations: [layerDom] string,
         trained: bool = false;

     proc init(dims: [] int, activations: [] string) {
       this.layerDom = {1..dims.size - 1};
       var layers: [layerDom] Layer;
       var caches: [layerDom] Cache;
       this.layers = layers;
       this.caches = caches;
       for l in layerDom {
         this.layers[l] = new Layer(activation = activations[l], udim = dims[l+1], ldim = dims[l]);
       }
     }

/*  Sends input data through a forwardPass of the Neural Network  */
     proc forwardPass(X:[]) {
       var Adom = X.domain;
       var A: [Adom] real = X;
       for l in this.layerDom {
        // const A_prev = A;
         const Z = this.layers[l].linearForward(A);
         const A_current = this.layers[l].activationForward(Z);
         if ! this.trained {
           this.caches[l] = new Cache();
           this.caches[l].aDom = A.domain;
           this.caches[l].A_prev = A;
           this.caches[l].zDom = Z.domain;
           this.caches[l].Z = Z;
         }
         Adom = A_current.domain;
         A = A_current;
       }
       return A;
     }
   }

   class Cache {
     var bDom: domain(1),
         wDom: domain(2),
         aDom: domain(2),
         zDom: domain(2),
         A_prev:[aDom] real,
         Z:[zDom] real,
         dW:[wDom] real,
         db:[bDom] real,
         dA_prev: [aDom] real,
         dZ:[zDom] real;

     proc init() { }
   }

/*  A Layer of a Neural Network is defined by it's activation, weights, and bias  */
   class Layer {
     var wDom: domain(2),
         bDom: domain(1),
         W:[wDom] real,
         b:[bDom] real,
         g: Activation;

/*  Constructs a layer with given activation and weights/bias initialized
         with small random postive numbers                                */
     proc init(activation: string, udim: int, ldim: int, eps = 0.1) {
       this.wDom = {1..udim,1..ldim};
       this.bDom = this.wDom.dim(1);
       var W: [wDom] real;
       var b: [bDom] real;
       this.W = W;
       this.b = b;
       fillRandom(this.W);
       fillRandom(this.b);
       this.W = eps*this.W;
       this.b = eps*this.b;
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


/*  Computes an Affine Transformation on A_prev:  Z = W.A_prev + b  */
     proc linearForward(A_prev: []) {
       const zDom: domain(2) = {this.W.domain.dim(1),A_prev.domain.dim(2)};
       var b: [zDom] real;  // this step is essentially python's "broadcasting"
       for i in zDom.dim(2) {  // this could be a `forall` i think
         b[..,i] = this.b;
       }
       const Z = this.W.dot(A_prev).plus(b);
       return Z;
     }

/*  Compute the activation of the current layer  */
     proc activationForward(Z: []) {
       const A: [Z.domain] real = this.g.f(Z);
       return A;
     }

     proc activationBackward(dA:[],Z:[]) {

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
       } else if this.name == "linear" {
         return x;
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
         return dheaviside(x);  //maybe I'll make this dsigmoid(x) for fun?
       } else if this.name == "linear" {
         return 1;
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

     proc id(x) {
       return x;
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

     proc did(x) {
       return 1;
     }
  }

  proc computeCost(Y:[], AL:[]) {
    var Jp: [AL.domain] real = Y*log(AL) + (1-Y)*log(1-AL);
    var J: real = -(+ reduce Jp)/AL.domain.dim(2).size;
    return J;
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
