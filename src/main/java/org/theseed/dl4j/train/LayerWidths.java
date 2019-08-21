/**
 *
 */
package org.theseed.dl4j.train;

/**
 * This class is used to compute layer widths in a model.  The constructor takes as
 * input the number of channels and the number of input columns.  For each type of
 * layer, it stores the input and output widths, which can be interrogated by the
 * client.
 *
 * @author Bruce Parrello
 *
 */
public class LayerWidths {

    // FIELDS
    /** current input width */
    private int inWidth;
    /** current output width */
    private int outWidth;
    /** current input height */
    private int channels;

    /**
     * Initialize the layer widths object.
     *
     * @param inputs	number of input columns
     * @param channels	number of values per column
     */
    public LayerWidths(int inputs, int channels) {
        this.inWidth = inputs;
        this.outWidth = inputs;
        this.channels = channels;
    }

    /**
     * @return the current input width
     */
    public int getInWidth() {
        return this.inWidth;
    }

    /**
     * @return the current output width
     */
    public int getOutWidth() {
        return this.outWidth;
    }

    /**
     * @return the number of channels
     */
    public int getChannels() {
        return this.channels;
    }

    /**
     * Compute the output from adding a convolution layer.
     *
     * @param kernel	kernel size
     * @param stride	stride between kernels
     * @param filters	number of filters per kernel
     */
    public void applyConvolution(int kernel, int stride, int filters) {
        this.inWidth = this.outWidth;
        this.outWidth = (this.inWidth - kernel) / stride + 1;
        // The new channel size is the number of filters.
        this.channels = filters;
    }

    /**
     * Compute the output from adding a subsampling layer.
     *
     * @param subFactor		subsampling factor
     */
    public void applySubsampling(int subFactor) {
        this.inWidth = this.outWidth;
        this.outWidth = (this.inWidth - subFactor) / subFactor + 1;
        // Note the channel size does not change
    }

    /**
     * Flatten the input for a feed-forward layer.
     */
    public void flatten() {
        this.outWidth = this.outWidth * this.channels;
        this.channels = 1;
    }

    /**
     * Compute the output from a feed-forward layer.
     *
     * @param layerSize	proposed number of output nodes
     */
    public void applyFeedForward(int layerSize) {
        this.inWidth = this.outWidth;
        this.outWidth = layerSize;
    }

    /**
     * @return 	an array containing a balanced layer configuration for the specified
     * 			number of layers producing the specified number of outputs
     *
     * @param layers		desired number of hidden layers
     * @param outputSize	number of output classes
     *
     */
    public int[] balancedLayers(int layers, int outputSize) {
        int[] retVal = new int[layers];
        int slope = (this.outWidth - outputSize) / (layers + 1);
        int extras = (this.outWidth - outputSize) % (layers + 1);
        int width = this.outWidth;
        for (int i = 0; i < layers; i++) {
            width -= slope + (extras > 0 ? 1 : 0);
            retVal[i] = width;
            extras--;
        }
        return retVal;
    }


}
