# InvoiceAgent — Schema Documentation

This document describes every Pydantic model in the InvoiceAgent pipeline.
Models are defined in `src/models/` and serve as the contracts between pipeline stages.

---

## Pipeline Data Flow

```
IngestResult → VisionExtractionResult → StructuredInvoice → QAResult → DeliveryResult
```

---

## Extraction Models (`src/models/extraction.py`)

### SourceRegion
Traces an extracted value back to its location in the source document.

| Field          | Type   | Required | Description                              |
|----------------|--------|----------|------------------------------------------|
| `page_number`  | int    | Yes      | 1-indexed page where the data was found  |
| `text_snippet` | str    | Yes      | Verbatim text from the source region     |

### PageImage
A single page image extracted from an ingested document.

| Field         | Type      | Required | Default | Description                     |
|---------------|-----------|----------|---------|---------------------------------|
| `page_number` | int       | Yes      | —       | 1-indexed page number           |
| `image_path`  | str       | Yes      | —       | Path to the extracted page image|
| `width`       | int\|None | No       | None    | Image width in pixels           |
| `height`      | int\|None | No       | None    | Image height in pixels          |

### IngestResult
Result of ingesting a raw document (PDF, image) into the pipeline.

| Field         | Type            | Required | Default          | Description                        |
|---------------|-----------------|----------|------------------|------------------------------------|
| `ingest_id`   | UUID            | No       | auto-generated   | Unique identifier for this ingest  |
| `source_file` | str             | Yes      | —                | Original file path or URI          |
| `ingested_at` | datetime        | No       | now (UTC)        | Timestamp of ingestion             |
| `page_count`  | int             | Yes      | —                | Total pages in the document        |
| `pages`       | list[PageImage] | No       | []               | Extracted page images              |
| `mime_type`   | str             | No       | application/pdf  | MIME type of the source file       |

### VisionExtractionResult
Raw extraction output from the Claude vision agent.

| Field            | Type               | Required | Default | Description                              |
|------------------|--------------------|----------|---------|------------------------------------------|
| `ingest_id`      | UUID               | Yes      | —       | References the IngestResult              |
| `extracted_at`   | datetime           | No       | now     | Timestamp of extraction                  |
| `vendor_name`    | str\|None          | No       | None    | Vendor / supplier name                   |
| `vendor_address` | str\|None          | No       | None    | Vendor address                           |
| `invoice_number` | str\|None          | No       | None    | Invoice number                           |
| `invoice_date`   | date\|None         | No       | None    | Invoice date                             |
| `due_date`       | date\|None         | No       | None    | Payment due date                         |
| `currency`       | str                | No       | USD     | ISO 4217 currency code                   |
| `total`          | Decimal\|None      | No       | None    | Invoice total amount                     |
| `subtotal`       | Decimal\|None      | No       | None    | Subtotal before tax                      |
| `tax`            | Decimal\|None      | No       | None    | Tax amount                               |
| `raw_line_items` | list[dict]         | No       | []      | Line items as raw dicts                  |
| `source_regions` | list[SourceRegion] | No       | []      | Source regions for traceability           |
| `raw_text`       | str\|None          | No       | None    | Full raw text from vision                |

---

## Invoice Models (`src/models/invoice.py`)

### InvoiceMetadata
Metadata about an invoice's processing journey.

| Field              | Type     | Required | Default | Description                          |
|--------------------|----------|----------|---------|--------------------------------------|
| `source_file`      | str      | Yes      | —       | Original file path or URI            |
| `ingest_id`        | UUID     | Yes      | —       | UUID from the ingestion stage        |
| `processed_at`     | datetime | No       | now     | When the invoice was structured      |
| `page_count`       | int      | No       | 1       | Number of pages in source document   |
| `pipeline_version` | str      | No       | 1.0.0   | Pipeline version that produced this  |

### LineItem
A single line item on an invoice.

| Field         | Type          | Required | Default | Description                        |
|---------------|---------------|----------|---------|------------------------------------|
| `description` | str           | Yes      | —       | Item description                   |
| `quantity`    | Decimal       | Yes      | —       | Quantity ordered/delivered          |
| `unit_price`  | Decimal       | Yes      | —       | Price per unit                     |
| `amount`      | Decimal       | Yes      | —       | Line total (quantity x unit_price)  |
| `sku`         | str\|None     | No       | None    | SKU or product code                |
| `tax`         | Decimal\|None | No       | None    | Tax amount for this line           |
| `gl_code`     | str\|None     | No       | None    | General ledger account code        |
| `source_page` | int           | Yes      | —       | Page number where item was found   |

### StructuredInvoice
The canonical output of the structuring agent — a fully structured, validated invoice.

| Field            | Type            | Required | Default        | Description                       |
|------------------|-----------------|----------|----------------|-----------------------------------|
| `invoice_id`     | UUID            | No       | auto-generated | Unique invoice identifier         |
| `invoice_number` | str             | Yes      | —              | Invoice number from the document  |
| `invoice_date`   | date            | Yes      | —              | Date on the invoice               |
| `due_date`       | date\|None      | No       | None           | Payment due date                  |
| `vendor_name`    | str             | Yes      | —              | Vendor / supplier name            |
| `vendor_address` | str\|None       | No       | None           | Vendor address                    |
| `currency`       | str             | No       | USD            | ISO 4217 currency code            |
| `subtotal`       | Decimal\|None   | No       | None           | Subtotal before tax               |
| `tax`            | Decimal\|None   | No       | None           | Total tax amount                  |
| `total`          | Decimal         | Yes      | —              | Invoice total (Decimal, not float)|
| `line_items`     | list[LineItem]  | No       | []             | Invoice line items                |
| `metadata`       | InvoiceMetadata | Yes      | —              | Processing metadata               |

---

## QA Models (`src/models/qa.py`)

### IssueType (Enum)
Categories of QA issues.

| Value            | Description                              |
|------------------|------------------------------------------|
| `missing`        | Required field was not extracted          |
| `low_confidence` | Extraction confidence below threshold    |
| `inconsistent`   | Field conflicts with other data          |

### QAFlag
A single quality issue flagged during QA.

| Field        | Type      | Required | Description                               |
|--------------|-----------|----------|-------------------------------------------|
| `field_name` | str       | Yes      | The field that has an issue               |
| `issue_type` | IssueType | Yes      | Type of issue detected                    |
| `message`    | str       | Yes      | Human-readable description of the issue   |

### FieldScore
Confidence score for a single extracted field.

| Field         | Type      | Required | Default | Description                        |
|---------------|-----------|----------|---------|------------------------------------|
| `field_name`  | str       | Yes      | —       | Name of the scored field           |
| `confidence`  | float     | Yes      | —       | Confidence score 0.0–1.0          |
| `source_page` | int\|None | No       | None    | Page the field was extracted from  |

### QAResult
Output of the QA agent — confidence scores and flags for a structured invoice.

| Field                | Type             | Required | Default | Description                         |
|----------------------|------------------|----------|---------|-------------------------------------|
| `invoice_id`         | UUID             | Yes      | —       | UUID of the StructuredInvoice       |
| `reviewed_at`        | datetime         | No       | now     | When the QA review was performed    |
| `overall_confidence` | float            | Yes      | —       | Overall confidence 0.0–1.0         |
| `field_scores`       | list[FieldScore] | No       | []      | Per-field confidence scores         |
| `flags`              | list[QAFlag]     | No       | []      | Quality issues found                |
| `approved`           | bool             | No       | False   | Whether the invoice passed QA       |

---

## Delivery Models (`src/models/delivery.py`)

### DeliveryFormat (Enum)
Supported output formats.

| Value  | Description       |
|--------|-------------------|
| `csv`  | CSV file output   |
| `json` | JSON file output  |

### DeliveryResult
Result of delivering a processed invoice to its destination.

| Field           | Type           | Required | Default | Description                          |
|-----------------|----------------|----------|---------|--------------------------------------|
| `invoice_id`    | UUID           | Yes      | —       | UUID of the delivered invoice        |
| `delivered_at`  | datetime       | No       | now     | When the delivery completed          |
| `format`        | DeliveryFormat | Yes      | —       | Output format used                   |
| `output_path`   | str            | Yes      | —       | Path or URI where output was written |
| `success`       | bool           | No       | True    | Whether delivery succeeded           |
| `error_message` | str\|None      | No       | None    | Error details if delivery failed     |
| `record_count`  | int            | No       | 0       | Number of line items delivered       |
