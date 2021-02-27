/**
 *
 */
package org.theseed.dl4j;

/**
 * This class is used to filter input columns for the DistributedOutputStream.  The default
 * subclass simply allows all columns to the output.  Other filters will need to override
 * "allows" to specify which columns to keep.
 *
 * @author Bruce Parrello
 *
 */
public abstract class BalanceColumnFilter {

    /**
     * @return TRUE if the specified column should be included in the output, else FALSE
     *
     * @param string	name of the relevant column
     */
    public abstract boolean allows(String string);

    public static class All extends BalanceColumnFilter {

        @Override
        public boolean allows(String string) {
            return true;
        }

    }

}
