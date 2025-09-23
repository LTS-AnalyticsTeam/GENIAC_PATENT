"""
Filter patent CSV data based on Cosmos DB reference.syutugan field
"""
import csv
import logging
import os
import sys
from datetime import datetime
from typing import Dict

from azure.cosmos import CosmosClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatentCSVFilter:
    """Filter patent CSV data based on Cosmos DB reference data"""

    def __init__(self, input_csv_path: str, output_csv_path: str):
        """
        Initialize the filter with input and output file paths

        Args:
            input_csv_path: Path to input CSV file
            output_csv_path: Path to output filtered CSV file
        """
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.cosmos_client = None
        self.reference_syutugan_set = set()

        # Statistics
        self.total_rows = 0
        self.passed_rows = 0
        self.failed_ax_docs = 0
        self.failed_ay_docs = 0
        self.failed_syutugan = 0

        # Stage-wise pass counts
        self.passed_syutugan = 0
        self.passed_ax_docs = 0
        self.passed_ay_docs = 0

    def connect_cosmos_db(self):
        """Connect to Cosmos DB and retrieve reference.syutugan data"""
        try:
            # Get Cosmos DB credentials from environment
            endpoint = os.getenv("COSMOS_ENDPOINT")
            key = os.getenv("COSMOS_KEY")
            database_name = os.getenv("COSMOS_DATABASE", "patent_db")
            container_name = os.getenv("COSMOS_CONTAINER", "patents")

            if not endpoint or not key:
                raise ValueError("COSMOS_ENDPOINT and COSMOS_KEY must be set in environment variables")

            logger.info(f"Connecting to Cosmos DB: {database_name}/{container_name}")

            # Initialize Cosmos client
            client = CosmosClient(endpoint, key)
            database = client.get_database_client(database_name)
            container = database.get_container_client(container_name)

            # Query to get all reference.syutugan values
            query = """
            SELECT DISTINCT c.metadata.reference.syutugan
            FROM c
            WHERE IS_DEFINED(c.metadata.reference.syutugan)
            AND c.metadata.reference.exist = true
            """

            logger.info("Fetching reference.syutugan data from Cosmos DB...")

            items = container.query_items(
                query=query,
                enable_cross_partition_query=True
            )

            # Collect all syutugan values
            for item in items:
                syutugan_list = item.get('syutugan', [])
                if syutugan_list:
                    for patent_id in syutugan_list:
                        if patent_id:  # Skip empty values
                            self.reference_syutugan_set.add(patent_id)

            logger.info(f"Loaded {len(self.reference_syutugan_set)} unique reference.syutugan values from Cosmos DB")

        except Exception as e:
            logger.error(f"Error connecting to Cosmos DB: {e}")
            raise

    def check_patent_match(self, patent_id: str) -> bool:
        """
        Check if a patent ID exists in the reference.syutugan set

        Args:
            patent_id: Patent ID to check

        Returns:
            True if patent ID exists in reference set, False otherwise
        """
        return patent_id in self.reference_syutugan_set

    def process_row(self, row: Dict[str, str]) -> bool:
        """
        Process a single CSV row through the two-stage check
        Order: syutugan -> ax_docs (ay_docs check is skipped)

        Args:
            row: CSV row as dictionary

        Returns:
            True if row passes syutugan and ax_docs checks, False otherwise
        """
        # Stage 1: Check syutugan FIRST
        syutugan = row.get('syutugan', '').strip()
        if syutugan:
            if not self.check_patent_match(syutugan):
                self.failed_syutugan += 1
                logger.debug(f"Row {row['case_id']} failed syutugan check: {syutugan}")
                return False

        # Passed syutugan check
        self.passed_syutugan += 1

        # Stage 2: Check ax_docs
        ax_docs = row.get('ax_docs', '').strip()
        if ax_docs:
            if not self.check_patent_match(ax_docs):
                self.failed_ax_docs += 1
                logger.debug(f"Row {row['case_id']} failed ax_docs check: {ax_docs}")
                return False

        # Passed ax_docs check - this is the final check
        self.passed_ax_docs += 1
        return True

    def filter_csv(self):
        """Main method to filter the CSV file"""
        try:
            # First, connect to Cosmos DB and load reference data
            self.connect_cosmos_db()

            if not self.reference_syutugan_set:
                logger.warning("No reference.syutugan data found in Cosmos DB. No rows will pass the filter.")

            # Open input and output CSV files
            with open(self.input_csv_path, 'r', encoding='utf-8') as infile, \
                 open(self.output_csv_path, 'w', encoding='utf-8', newline='') as outfile:

                reader = csv.DictReader(infile)
                fieldnames = reader.fieldnames

                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                logger.info(f"Processing CSV file: {self.input_csv_path}")

                # Process each row
                for row in reader:
                    self.total_rows += 1

                    if self.process_row(row):
                        writer.writerow(row)
                        self.passed_rows += 1
                        logger.info(f"Row {row['case_id']} passed all checks")

                    # Progress update every 100 rows
                    if self.total_rows % 100 == 0:
                        logger.info(f"Processed {self.total_rows} rows, {self.passed_rows} passed")

            # Print summary
            self.print_summary()

        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise

    def print_summary(self):
        """Print processing summary"""
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY (Order: syutugan → ax_docs)")
        logger.info("=" * 60)
        logger.info(f"Total rows processed: {self.total_rows}")
        logger.info("")
        logger.info("Stage-wise filtering results:")

        # Calculate percentages
        syutugan_pct = self.passed_syutugan/self.total_rows*100
        ax_docs_pct = self.passed_ax_docs/self.total_rows*100
        final_pct = self.passed_rows/self.total_rows*100

        logger.info(f"1. After syutugan check: {self.passed_syutugan} rows "
                    f"remaining ({syutugan_pct:.1f}%)")
        logger.info(f"   - Failed: {self.failed_syutugan} rows")
        logger.info(f"2. After ax_docs check: {self.passed_ax_docs} rows "
                    f"remaining ({ax_docs_pct:.1f}%)")
        logger.info(f"   - Failed: {self.failed_ax_docs} rows")
        logger.info("")
        logger.info(f"Final result: {self.passed_rows} rows passed "
                    f"all checks ({final_pct:.2f}%)")
        logger.info(f"Output file: {self.output_csv_path}")
        logger.info("=" * 60)


def main():
    """Main function"""
    # Set file paths
    input_csv = "test_cases.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"filtered_test_cases_{timestamp}.csv"

    # Check if input file exists
    if not os.path.exists(input_csv):
        logger.error(f"Input file not found: {input_csv}")
        sys.exit(1)

    # Create filter instance and run
    filter = PatentCSVFilter(input_csv, output_csv)

    try:
        filter.filter_csv()
        logger.info(f"Filtering completed successfully. Output saved to: {output_csv}")
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
