/**
 *
 */
package org.theseed.dl4j;

import java.util.Arrays;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.linalg.api.shape.Shape;

/**
 * This class converts a time series suitable for a recurrent neural network layer to a flat layer
 * for feed-forward processing.  The input has shape [batchSize, channels, width] and the
 * output has shape [batchSize, channels * width].
 *
 * @author Bruce Parrello
 *
 */
public class RnnSequenceToFeedForwardPreProcessor implements InputPreProcessor {

    // FIELDS
    private long numChannels;
    private long inputWidth;

    /**
     * serialization code number
     */
    private static final long serialVersionUID = 1052070337217338101L;

    @JsonCreator
    public RnnSequenceToFeedForwardPreProcessor(@JsonProperty("numChannels") long numChannels,
            @JsonProperty("inputWidth") long inputWidth) {
        this.numChannels = numChannels;
        this.inputWidth = inputWidth;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (input.rank() != 3)
            throw new IllegalArgumentException(
                            "Invalid input: expect RNN activations with rank 3 (received input with shape "
                                            + Arrays.toString(input.shape()) + ")");
        if (input.size(1) != this.numChannels || input.size(2) != this.inputWidth)
            throw new IllegalArgumentException(
                    "Invalid input: expect RNN activations with channel size " + this.numChannels
                            + " and width " + this.inputWidth + " (received input with shape "
                            + Arrays.toString(input.shape()) + ")");
        //Input: 3d activations (RNN)
        //Output: 2d activations (FF)

        if (input.ordering() != 'c' || !Shape.hasDefaultStridesForShape(input))
            input = input.dup('c');

        // Combine the channels and the time sequence.
        INDArray retVal = input.reshape('c', input.size(0), input.size(1) * input.size(2));
        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, retVal);
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        //Output: 2d activations (RNN)
        //Input: 3d activations (CNN)
        INDArray retVal = output.reshape('c', output.size(0), this.numChannels, this.inputWidth);
        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, retVal);
    }

    @Override
    public InputPreProcessor clone() {
        return new RnnSequenceToFeedForwardPreProcessor(this.numChannels, this.inputWidth);
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input type: Expected input of type RNN, got " + inputType);
        }

        return InputType.feedForward(this.numChannels * this.inputWidth);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
            int minibatchSize) {
        //Assume mask array is 4d - a mask array that has been reshaped from [minibatch,timeSeriesLength] to [minibatch*timeSeriesLength, 1, 1, 1]
        if (maskArray == null) {
            return new Pair<>(maskArray, currentMaskState);
        } else {
            //Need to reshape mask array from [minibatch*timeSeriesLength, 1, 1, 1] to [minibatch,timeSeriesLength]
            return new Pair<>(TimeSeriesUtils.reshapeCnnMaskToTimeSeriesMask(maskArray, minibatchSize),currentMaskState);
        }
    }

}
