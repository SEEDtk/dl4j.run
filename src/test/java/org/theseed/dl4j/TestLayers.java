/**
 *
 */
package org.theseed.dl4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.jupiter.api.Test;
import org.theseed.dl4j.train.LayerWidths;

/**
 * @author Bruce Parrello
 *
 */
class TestLayers {

    /**
     * Test the layer widths computer
     */
    @Test
    public void testLayerWidths() {
        LayerWidths widthComputer = new LayerWidths(50, 4);
        assertThat(widthComputer.getInWidth(), equalTo(50));
        assertThat(widthComputer.getOutWidth(), equalTo(50));
        assertThat(widthComputer.getChannels(), equalTo(4));
        widthComputer.applyConvolution(3, 2, 10);
        assertThat(widthComputer.getInWidth(), equalTo(50));
        assertThat(widthComputer.getOutWidth(), equalTo(24));
        assertThat(widthComputer.getChannels(), equalTo(10));
        widthComputer.applySubsampling(2);
        assertThat(widthComputer.getInWidth(), equalTo(24));
        assertThat(widthComputer.getOutWidth(), equalTo(12));
        assertThat(widthComputer.getChannels(), equalTo(10));
        widthComputer.flatten();
        assertThat(widthComputer.getOutWidth(), equalTo(120));
        assertThat(widthComputer.getChannels(), equalTo(1));
        Integer[] layers = ArrayUtils.toObject(widthComputer.balancedLayers(4, 3));
        assertThat(layers, arrayContaining(96, 72, 49, 26));
        widthComputer.applyFeedForward(96);
        assertThat(widthComputer.getInWidth(), equalTo(120));
        assertThat(widthComputer.getOutWidth(), equalTo(96));
    }
}
