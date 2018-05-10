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

  }

  proc testHiddenLayer() {


    var udim = 3,
        ldim = 4,
        activation = "relu";

    var layer = new Layer(activation, udim, ldim);
    var reluP = layer.g.f(3);
    var reluN = layer.g.f(-3);

    assertRealApproximates("RELU Activation is Correct",expected=3,actual=reluP+reluN);

  }

  proc testLinearForward() {


    var X = Matrix( [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0] );
    var udim = 2,
        ldim = X.shape[1],
        activation = "relu";

    var layer = new Layer(activation, udim, ldim);

    var Z = layer.linearForward(X);
    assertIntEquals("Top Dim is W.shape[1]",expected=layer.W.shape[1],actual=Z.shape[1]);
    assertIntEquals("Bottom Dim is X.shape[2]",expected=X.shape[2],actual=Z.shape[2]);

  }

  proc testActivationForward() {


    var X = Matrix( [100.0,0.0,10.0,0.0], [100.0,0.0,10.0,0.0], [100.0,0.0,10.0,0.0] );
    var udim = 2,
        ldim = X.shape[1],
        activation = "relu";

    var layer = new Layer(activation, udim, ldim);

    var Z = layer.linearForward(X);
    var A = layer.activationForward(Z);

    assertRealApproximates("In this test, A equals Z", expected=Z[1,1], actual=A[1,1]);

  }

  proc testStackBuilder() {


    var dims = [3,4,2,3,1],
        activations = ["sigmoid","sigmoid","sigmoid","sigmoid"];

    var model = new FCNetwork(dims, activations);
    assertIntEquals("Output Dimension of Layer 3 should be 3", expected=3, actual=model.layers[3].W.shape(1));

  }

  proc testForwardPass() {


    var dims = [3,4,2,3,1],
        activations = ["sigmoid","sigmoid","sigmoid","sigmoid"];

    var X = Matrix( [100000.0,0.0,10.0,0.0], [10000.0,0.0,10.0,0.0], [1000.0,0.0,10.0,0.0] );

    var model = new FCNetwork(dims, activations);

    model.trained = true;

    var output = model.forwardPass(X);
    assertIntEquals("Single Node Output Expected", expected=1, actual=output.shape(1));

  }

  proc testCostFunction() {


    var dims = [2,2,1],
        activations = ["tanh","sigmoid"];

    var model = new FCNetwork(dims, activations);

    var X = Matrix( [10.0, 0.0, 10.0, 0.0],
                    [10.0, 0.0, 10.0, 0.0]);

    var Y = Matrix( [1.0, 1.0, 1.0, 1.0] );

    var AL = model.forwardPass(X);

    var cost = computeCost(Y,AL);

    assertBoolEquals("Cost Should be Higher Than 0.6",expected=true,actual=cost>0.6);

  }

  proc testLinearBackward() {


    var dZ = Matrix( [1.0,0.0], [0.0,1.0] );
    var A_prev = Matrix( [0.0,1.0], [1.0,0.0] );

    var layer = new Layer(activation = "linear" , udim = 2, ldim = 2);

    var (dW, db, dA_prev) = layer.linearBackward(dZ, A_prev);

    assertRealApproximates("Sum over dW",expected=1,actual=(+ reduce dW));

  }

  proc testXOR() {


    var t: Timer;
    t.start();



    var X = Matrix( [0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0] );

    var Y = Matrix( [0.0, 1.0, 1.0, 0.0] );

    var dims = [X.shape[1],2,1],  // 2d inputs, 2-unit layer, 1d output
        activations = ["tanh","linear"],
        numEpochs = 100000,
        reportInterval = 1000,
        learningRate = 0.01;

    var testX = Matrix( [0.0, 0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0, 0.0] );

    var testY = Matrix( [1.0, 0.0, 0.0, 1.0] );


    var model = new FCNetwork(dims,activations);

    model.train_(X,Y,numEpochs,learningRate,reportInterval);

    var preds = model.forwardPass(testX);

    var cost = model.loss.J(testY,preds);

    assertBoolEquals("Cost less than 0.00000001", expected=true, actual=0.000000001>cost);
  }

  proc run() {
    super.run();
    testBreath();
    testHiddenLayer();
    testLinearForward();
    testActivationForward();
    testStackBuilder();
    testForwardPass();
    testCostFunction();
    testLinearBackward();
    testXOR();
    return 0;
  }
}

proc main(args: [] string) : int {
  var t = new EpochTest(verbose=false);
  var ret = t.run();
  t.report();
  return ret;
}
