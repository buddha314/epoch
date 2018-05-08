use NumSuch,
    Norm,
    LinearAlgebra,
    Viterbi,
    Epoch,
    Math,
    Core,
    Neural,
    Charcoal;


class EpochTest : UnitTest {
  var nv: int = 8,
      D: domain(2) = {1..nv, 1..nv},
      SD: sparse subdomain(D),
      X: [SD] real;


  var vn: [1..0] string;

  proc setUp() {
    vn.push_back("star lord");
    vn.push_back("gamora");
    vn.push_back("groot");
    vn.push_back("drax");
    vn.push_back("rocket");
    vn.push_back("mantis");
    vn.push_back("yondu");
    vn.push_back("nebula");

    SD += (1,2); X[1,2] = 1;
    SD += (1,3); X[1,3] = 1;
    SD += (1,4); X[1,4] = 1;
    SD += (2,2); X[2,2] = 1;
    SD += (2,4); X[2,4] = 1;
    SD += (3,4); X[3,4] = 1;
    SD += (4,5); X[4,5] = 1;
    SD += (5,6); X[5,6] = 1;
    SD += (6,7); X[6,7] = 1;
    SD += (6,8); X[6,8] = 1;
    SD += (7,8); X[7,8] = 1;
  }


  proc init(verbose:bool) {
    super.init(verbose=verbose);
    this.complete();
  }

  proc testBreath(){
    writeln("");
    writeln("");
    writeln("Breathe Deep...");
    writeln("");
    writeln("");
    writeln("");
  }

  proc testHiddenLayer() {
    writeln("testHiddenLayer... starting...");
    writeln("");

    var udim = 3,
        ldim = 4,
        activation = "relu";

    var layer = new Layer(activation, udim, ldim);


    writeln("Initialized Weight Matrix: ");
    writeln(layer.W);
    writeln("Shape of W: ",layer.W.shape);
    writeln("");
    writeln("Initialized Bias Vector: ");
    writeln(layer.b);
    writeln("Shape of b: ",layer.b.shape);

    var reluP = layer.g.f(3);
    var reluN = layer.g.f(-3);

    writeln("");
    writeln("Checking Activation: relu(3)= ",reluP," and relu(-3)= ",reluN);

    assertRealApproximates("RELU Activation is Correct",expected=3,actual=reluP+reluN);

    writeln("");
    writeln("testHiddenLayer... done...");
    writeln("");
    writeln("");
  }

  proc testLinearForward() {
    writeln("");
    writeln("");
    writeln("testLinearForward... starting...");
    writeln("");

    var X = Matrix( [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0] );
    var udim = 2,
        ldim = X.shape[1],
        activation = "relu";

    var layer = new Layer(activation, udim, ldim);

    var Z = layer.linearForward(X);
    writeln(Z);
    writeln("Shapes: Z.shape = ",Z.shape,", W.shape = ",layer.W.shape,", X.shape = ",X.shape);

    assertIntEquals("Top Dim is W.shape[1]",expected=layer.W.shape[1],actual=Z.shape[1]);
    assertIntEquals("Bottom Dim is X.shape[2]",expected=X.shape[2],actual=Z.shape[2]);

    writeln("");
    writeln("testLinearForward... done...");
    writeln("");
    writeln("");
  }

  proc testActivationForward() {
    writeln("");
    writeln("");
    writeln("testActivationForward... starting...");
    writeln("");

    var X = Matrix( [100.0,0.0,10.0,0.0], [100.0,0.0,10.0,0.0], [100.0,0.0,10.0,0.0] );
    var udim = 2,
        ldim = X.shape[1],
        activation = "relu";

    var layer = new Layer(activation, udim, ldim);

    var Z = layer.linearForward(X);
    writeln("Affine Transformation: ");
    writeln(Z);
    writeln("Shapes: Z.shape = ",Z.shape,", W.shape = ",layer.W.shape,", X.shape = ",X.shape);

    var A = layer.activationForward(Z);

    writeln("");
    writeln("This Layer's Activation: ");
    writeln(A);

    assertRealApproximates("In this test, A equals Z", expected=Z[1,1], actual=A[1,1]);

    writeln("");
    writeln("testActivationForward... done...");
    writeln("");
    writeln("");
  }

  proc testStackBuilder() {
    writeln("");
    writeln("");
    writeln("testStackBuilder... starting...");
    writeln("");

    var dims = [3,4,2,3,1],
        activations = ["sigmoid","sigmoid","sigmoid","sigmoid"];

    var model = new FCNetwork(dims, activations);

    writeln("Number of Layers: ",model.layers.size);
    writeln("Weights of 3rd Layer: \n",model.layers[3].W);
    writeln("Shape of 3rd Weight Matrix: ",model.layers[3].W.shape);
    writeln("Activation Type of 3rd Layer: ",model.layers[3].g.name);

    assertIntEquals("Output Dimension of Layer 3 should be 3", expected=3, actual=model.layers[3].W.shape(1));

    writeln("");
    writeln("testStackBuilder... done...");
    writeln("");
    writeln("");
  }

  proc testForwardPass() {
    writeln("");
    writeln("");
    writeln("testForwardPass... starting...");
    writeln("");

    var dims = [3,4,2,3,1],
        activations = ["sigmoid","sigmoid","sigmoid","sigmoid"];

    var X = Matrix( [100000.0,0.0,10.0,0.0], [10000.0,0.0,10.0,0.0], [1000.0,0.0,10.0,0.0] );

    var model = new FCNetwork(dims, activations);

    model.trained = true;
    if model.trained then writeln("Treating this as a Trained Model \n");

    var output = model.forwardPass(X);

    writeln("Output: ",output);

    assertIntEquals("Single Node Output Expected", expected=1, actual=output.shape(1));


    writeln("");
    writeln("testForwardPass... done...");
    writeln("");
    writeln("");
  }

  proc testCostFunction() {
    writeln("");
    writeln("");
    writeln("testCostFunction... starting...");
    writeln("");

    var dims = [2,2,1],
        activations = ["tanh","sigmoid"];

    var model = new FCNetwork(dims, activations);

    var X = Matrix( [10.0, 0.0, 10.0, 0.0],
                    [10.0, 0.0, 10.0, 0.0]);

    var Y = Matrix( [1.0, 1.0, 1.0, 1.0] );

    var AL = model.forwardPass(X);

    var cost = computeCost(Y,AL);
    writeln("Cost: ",cost);

    assertBoolEquals("Cost Should be Higher Than 0.6",expected=true,actual=cost>0.6);

    writeln("");
    writeln("testCostFunction... done...");
    writeln("");
    writeln("");
  }

  proc testCaches() {
    writeln("");
    writeln("");
    writeln("testCaches... starting...");
    writeln("");

    var dims = [2,2,1],
        activations = ["tanh","sigmoid"];

    var model = new FCNetwork(dims, activations);

    var X = Matrix( [10.0, 0.0, 10.0, 0.0],
                    [10.0, 0.0, 10.0, 0.0]);

    var Y = Matrix( [1.0, 1.0, 1.0, 1.0] );

    var AL = model.forwardPass(X);

    var cost = computeCost(Y,AL);
    writeln("Cost: ",cost);
    writeln("");
    writeln("Caches:");
    for l in model.layerDom {
      writeln("Layer ",l," A: \n",model.caches[l].A);
      writeln("");
      writeln("Layer ",l," Z: \n",model.caches[l].Z);
      writeln("");
      writeln("");
    }

    assertIntEquals("Dim 1 of Second Cache's Z", expected=1, actual=model.caches[2].Z.shape(1));

    writeln("");
    writeln("testCaches... done...");
    writeln("");
    writeln("");
  }

  proc testLinearBackward() {
    writeln("");
    writeln("");
    writeln("testLinearBackward... starting...");
    writeln("");

    var dZ = Matrix( [1.0,0.0], [0.0,1.0] );
    var A_prev = Matrix( [0.0,1.0], [1.0,0.0] );

    var layer = new Layer(activation = "linear" , udim = 2, ldim = 2);

    var (dW, db, dA_prev) = layer.linearBackward(dZ, A_prev);

    writeln(dW);
    writeln(dW.shape);
    writeln("");
    writeln(dA_prev);
    writeln(dA_prev.shape);
    writeln("");
    writeln(db);
    writeln(db.shape);
    writeln(db.domain);

    assertRealApproximates("Sum over dW",expected=1,actual=(+ reduce dW));

    writeln("");
    writeln("testLinearBackward... done...");
    writeln("");
    writeln("");
  }

  proc testBackProp() {
    writeln("");
    writeln("");
    writeln("testBackProp... starting...");
    writeln("");

    var dims = [2,2,2,1],
        activations = ["tanh","tanh","sigmoid"];

    var model = new FCNetwork(dims, activations);

    var X = Matrix( [10.0, 0.0, 10.0, 0.0],
                    [10.0, 0.0, 10.0, 0.0]);

    var Y = Matrix( [1.0, 1.0, 1.0, 1.0] );

    var AL = model.forwardPass(X);

    var cost = computeCost(Y,AL);
    writeln("Cost: ",cost);
    writeln("");
    writeln("Caches:");
    for l in model.layerDom {
      writeln("Layer ",l," W: \n",model.layers[l].W);
      writeln("");
      writeln("Layer ",l," b: \n",model.layers[l].b);
      writeln("");
      writeln("");
    }

    writeln("");

    model.backwardPass(AL, Y);

    for l in model.layerDom {
      writeln("Layer ",l," dW: \n",model.caches[l].dW);
      writeln("");
      writeln("Layer ",l," db: \n",model.caches[l].db);
      writeln("");
      writeln("");
    }

    writeln("Before Update: \n",model.caches);

    model.updateParameters(learningRate = 0.001);

    writeln("After Update: \n",model.caches);

    for l in model.layerDom {
      writeln("Layer ",l," W: \n",model.layers[l].W);
      writeln("");
      writeln("Layer ",l," b: \n",model.layers[l].b);
      writeln("");
      writeln("");
    }

    var AL2 = model.forwardPass(X);

    var cost2 = computeCost(Y,AL2);
    writeln("Cost: ",cost2);


  //  assertIntEquals("Dim 1 of Second Cache's Z", expected=1, actual=model.caches[2].Z.shape(1));

    writeln("");
    writeln("testBackProp... done...");
    writeln("");
    writeln("");
  }

  proc testXOR() {
    writeln("");
    writeln("");
    writeln("testXOR... starting...");
    writeln("");

    var t: Timer;
    t.start();


    var dims = [2,2,1],
        activations = ["tanh","sigmoid"],
        epochs=400000,
        reportInterval = 1000,
        learningRate = 0.01;

        var X = Matrix( [0.0, 0.0, 1.0, 1.0],
                        [0.0, 1.0, 0.0, 1.0]);

        var Y = Matrix( [0.0, 1.0, 1.0, 0.0] );

    var testData = X;

    var model = new FCNetwork(dims,activations);

    model.train(X,Y,epochs,learningRate,reportInterval);

    writeln("\n\n");

    var preds = model.forwardPass(testData);
    writeln("XOR Predictions: ",preds);
    writeln("");

    t.stop();
    writeln("Training took: ",t.elapsed()," seconds");


    writeln("");
    writeln("testXOR... done...");
    writeln("");
    writeln("");

  }

  proc testMiniBatching() {
    writeln("");
    writeln("");
    writeln("testMiniBatching... starting...");
    writeln("");

    var t: Timer;
    t.start();


    var dims = [2,2,1],
        activations = ["tanh","linear"],
        epochs=400000,
        reportInterval = 10000,
        batchsize = 4,
        learningRate = 0.01;

        var X = Matrix( [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        var Y = Matrix( [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0] );

    var testData = X;

    var model = new FCNetwork(dims,activations);

    model.train(X,Y,epochs,learningRate,reportInterval,batchsize);

    writeln("\n\n");

    var preds = model.forwardPass(testData);
    writeln("XOR Predictions: ",preds);
    writeln("");

    t.stop();
    writeln("Training took: ",t.elapsed()," seconds");


    writeln("");
    writeln("testMiniBatching... done...");
    writeln("");
    writeln("");
  }

  proc testSine() {
    writeln("");
    writeln("");
    writeln("testSine... starting...");
    writeln("");

    var t: Timer;
    t.start();


    var dom: domain(2) = {1..1,1..1000};
    var X,Z: [dom] real;
    fillRandom(X);
    fillRandom(Z);
    X = 2*pi*X;
    Z = 2*pi*Z;
    X = Matrix(X);
    Z = Matrix(Z);
    var Y: X.type = sin(X);
    var testX: Z.type = Z;
    var testY: Z.type = sin(testX);

    var dims = [X.shape[1],4,1],
        activations = ["tanh","linear"],
        epochs=100000,
        reportInterval = 10000,
        learningRate = 0.01;


    var model = new FCNetwork(dims,activations);

    model.train(X,Y,epochs,learningRate,reportInterval);

    writeln("\n\n");

    var preds = model.forwardPass(testX);
    writeln("Sine Predictions: ",preds[1,1..10]);
    writeln("Actual Values:    ",testY[1,1..10]);
    writeln("");

    t.stop();
    writeln("Training took: ",t.elapsed()," seconds");


    writeln("");
    writeln("testSine... done...");
    writeln("");
    writeln("");

  }

  proc run() {
    super.run();
//    testBreath();
//    testHiddenLayer();
//    testLinearForward();
//    testActivationForward();
//    testStackBuilder();
//    testForwardPass();
//    testCostFunction();
//    testCaches();
//    testLinearBackward();
//    testBackProp();
//    testXOR();
//    testMiniBatching();
    testSine();
    return 0;
  }
}

proc main(args: [] string) : int {
  var t = new EpochTest(verbose=false);
  var ret = t.run();
  t.report();
  return ret;
}
