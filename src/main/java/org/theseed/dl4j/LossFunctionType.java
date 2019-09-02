/**
 *
 */
package org.theseed.dl4j;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossCosineProximity;
import org.nd4j.linalg.lossfunctions.impl.LossFMeasure;
import org.nd4j.linalg.lossfunctions.impl.LossHinge;
import org.nd4j.linalg.lossfunctions.impl.LossKLD;
import org.nd4j.linalg.lossfunctions.impl.LossL1;
import org.nd4j.linalg.lossfunctions.impl.LossL2;
import org.nd4j.linalg.lossfunctions.impl.LossMAE;
import org.nd4j.linalg.lossfunctions.impl.LossMAPE;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.nd4j.linalg.lossfunctions.impl.LossMSLE;
import org.nd4j.linalg.lossfunctions.impl.LossPoisson;
import org.nd4j.linalg.lossfunctions.impl.LossSquaredHinge;

/**
 * This class manages loss functions.  It includes an enumeration for the loss function type and methods for
 * creating the function instance.
 *
 * @author Bruce Parrello
 *
 */
public enum LossFunctionType {

    XENT(true, "Binary Cross-Entropy"), COSINE_PROXIMITY(false, "Cosine Proximity"),
    FMEASURE(true, "F-Measure"), HINGE(false, "Hinge"), KLD(false, "Kullback-Leibler Divergence"),
    L1(false, "Absolute Error"), L2(false, "Squared Error"), MAE(false, "Mean Absolute Error"),
    MAPE(false, "Mean Absolute Percentage Error"), MCXENT(false, "Multi-Class Cross-Entropy"),
    MSE(false, "Mean Squared Error"), MSLE(false, "Mean Squared Logarithmic Error"),
    POISSON(false, "Poisson"), SQUARED_HINGE(false, "Squared Hinge");

    // FIELDS
    boolean binaryOnly;
    String label;

    /**
     * Construct a loss-function type.
     *
     * @param binaryOnly	TRUE if this type only works with binary classification
     * @param label			displayable name of the function
     */
    private LossFunctionType(boolean binaryOnly, String label) {
        this.binaryOnly = binaryOnly;
        this.label = label;
    }

    /**
     * @return the displayable name of this function
     */
    public String toString() {
        return this.label;
    }

    /**
     * @return TRUE if this function is binary-classification only
     */
    public boolean isBinaryOnly() {
        return this.binaryOnly;
    }

    /**
     * @return the appropriate output layer activation function for this loss function
     */
    public Activation getOutActivation() {
        return (this == XENT ? Activation.SIGMOID : Activation.SOFTMAX);
    }

    /**
     * @return an instance of a weighted loss function of this type
     *
     * @param weights	an array of weights, one per label
     */
    public ILossFunction create(double[] weights) {
        float[] weightCopy = new float[weights.length];
        for (int i = 0; i < weights.length; i++)
            weightCopy[i] = (float) weights[i];
        INDArray weightArray = Nd4j.create(weightCopy);
        ILossFunction retVal = null;
        switch (this) {
        case XENT:
            retVal = new LossBinaryXENT(weightArray);
            break;
        case COSINE_PROXIMITY:
            retVal = new LossCosineProximity();
            break;
        case FMEASURE:
            retVal = new LossFMeasure();
            break;
        case HINGE:
            retVal = new LossHinge();
            break;
        case KLD:
            retVal = new LossKLD();
            break;
        case L1:
            retVal = new LossL1(weightArray);
            break;
        case L2:
            retVal = new LossL2(weightArray);
            break;
        case MAE:
            retVal = new LossMAE(weightArray);
            break;
        case MAPE:
            retVal = new LossMAPE(weightArray);
            break;
        case MCXENT:
            retVal = new LossMCXENT(weightArray);
            break;
        case MSE:
            retVal = new LossMSE(weightArray);
            break;
        case MSLE:
            retVal = new LossMSLE(weightArray);
            break;
        case POISSON:
            retVal = new LossPoisson();
            break;
        case SQUARED_HINGE:
            retVal = new LossSquaredHinge();
            break;
        }
        return retVal;
    }
}
