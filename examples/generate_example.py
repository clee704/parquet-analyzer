from __future__ import annotations

import argparse
import math
import uuid
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_ROW_COUNT = 100_000
DEFAULT_ROW_GROUPS = 3
DEFAULT_SEED = 1337

HUNDRED = Decimal("100")

FIRST_NAMES = np.array(
    [
        "Alex",
        "Jordan",
        "Taylor",
        "Morgan",
        "Riley",
        "Casey",
        "Harper",
        "Jamie",
        "Drew",
        "Sydney",
    ],
    dtype=object,
)
LAST_NAMES = np.array(
    [
        "Lee",
        "Patel",
        "Nguyen",
        "Garcia",
        "Smith",
        "Khan",
        "Chen",
        "Silva",
        "Brown",
        "Ivanov",
    ],
    dtype=object,
)
LOYALTY_LEVELS = ("Bronze", "Silver", "Gold", "Platinum", "Diamond")
LOYALTY_PROBS = (0.25, 0.3, 0.22, 0.18, 0.05)
SEGMENT_BY_LOYALTY = {
    "Bronze": "Consumer",
    "Silver": "SMB",
    "Gold": "SMB",
    "Platinum": "Enterprise",
    "Diamond": "Enterprise",
}
SALES_CHANNELS = ("Online", "Retail", "Field", "Marketplace", "Partner")
STATUS_OPTIONS = ("Processing", "Shipped", "Delivered", "Delayed", "Cancelled", "Returned")
STATUS_PROBS = (0.25, 0.32, 0.28, 0.07, 0.05, 0.03)
DISCOUNT_RATES = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
DISCOUNT_PROBS = (0.25, 0.22, 0.2, 0.15, 0.12, 0.06)
TAX_RATES = np.array([0.05, 0.07, 0.0825, 0.095])
TAX_RATE_PROBS = (0.35, 0.28, 0.22, 0.15)
PAYMENT_METHODS = ("Credit Card", "Invoice", "ACH", "Wire", "Digital Wallet")
PAYMENT_METHOD_PROBS = (0.4, 0.25, 0.12, 0.13, 0.1)
DOMAINS = np.array(("example.com", "business.org", "retailers.net", "supplychain.co", "globalgoods.io"))
DELIVERY_METHODS = np.array(("Ground", "Air", "Freight", "Courier"))
ORDER_TYPES = np.array(("Standard", "Backorder", "Drop-ship", "Subscription"))

LOCATIONS = (
    {
        "region": "North America",
        "region_code": "NA-US-CA",
        "country": "United States",
        "state": "California",
        "city": "San Francisco",
        "postal_code": "94107",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "currency": "USD",
        "warehouse": "SFO1",
    },
    {
        "region": "North America",
        "region_code": "NA-US-NY",
        "country": "United States",
        "state": "New York",
        "city": "New York",
        "postal_code": "10001",
        "latitude": 40.7128,
        "longitude": -74.006,
        "currency": "USD",
        "warehouse": "JFK1",
    },
    {
        "region": "North America",
        "region_code": "NA-CA-ON",
        "country": "Canada",
        "state": "Ontario",
        "city": "Toronto",
        "postal_code": "M5H",
        "latitude": 43.6532,
        "longitude": -79.3832,
        "currency": "CAD",
        "warehouse": "YYZ2",
    },
    {
        "region": "Europe",
        "region_code": "EU-DE-BE",
        "country": "Germany",
        "state": "Berlin",
        "city": "Berlin",
        "postal_code": "10117",
        "latitude": 52.52,
        "longitude": 13.405,
        "currency": "EUR",
        "warehouse": "BER2",
    },
    {
        "region": "Europe",
        "region_code": "EU-UK-LON",
        "country": "United Kingdom",
        "state": "Greater London",
        "city": "London",
        "postal_code": "EC1A",
        "latitude": 51.5074,
        "longitude": -0.1278,
        "currency": "GBP",
        "warehouse": "LHR1",
    },
    {
        "region": "Asia Pacific",
        "region_code": "APAC-AU-NSW",
        "country": "Australia",
        "state": "New South Wales",
        "city": "Sydney",
        "postal_code": "2000",
        "latitude": -33.8688,
        "longitude": 151.2093,
        "currency": "AUD",
        "warehouse": "SYD1",
    },
    {
        "region": "Asia Pacific",
        "region_code": "APAC-JP-KAN",
        "country": "Japan",
        "state": "Tokyo",
        "city": "Tokyo",
        "postal_code": "100-0005",
        "latitude": 35.6762,
        "longitude": 139.6503,
        "currency": "JPY",
        "warehouse": "HND2",
    },
    {
        "region": "Asia Pacific",
        "region_code": "APAC-SG",
        "country": "Singapore",
        "state": "Singapore",
        "city": "Singapore",
        "postal_code": "049483",
        "latitude": 1.2839,
        "longitude": 103.8514,
        "currency": "SGD",
        "warehouse": "SIN1",
    },
    {
        "region": "South Asia",
        "region_code": "SA-IN-KA",
        "country": "India",
        "state": "Karnataka",
        "city": "Bengaluru",
        "postal_code": "560001",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "currency": "INR",
        "warehouse": "BLR2",
    },
)

PRODUCT_CATALOG = (
    {"sku": "ELEC-1001", "category": "Electronics", "min_price": 15_000, "max_price": 85_000},
    {"sku": "APP-2003", "category": "Apparel", "min_price": 2_500, "max_price": 16_000},
    {"sku": "FURN-3007", "category": "Furniture", "min_price": 45_000, "max_price": 240_000},
    {"sku": "ACC-4102", "category": "Accessories", "min_price": 1_200, "max_price": 9_500},
    {"sku": "FOOD-5201", "category": "Grocery", "min_price": 500, "max_price": 4_800},
    {"sku": "SERV-6105", "category": "Services", "min_price": 8_000, "max_price": 45_000},
)

CUSTOMER_NAME_FIELDS = [pa.field("first", pa.string()), pa.field("last", pa.string())]
CUSTOMER_NAME_TYPE = pa.struct(CUSTOMER_NAME_FIELDS)
WEIGHTS_FIELDS = [pa.field("gross", pa.float32()), pa.field("net", pa.float32())]
WEIGHTS_TYPE = pa.struct(WEIGHTS_FIELDS)
CUSTOMER_PROFILE_FIELDS = [
    pa.field("loyalty_level", pa.string()),
    pa.field("signup_date", pa.date32()),
    pa.field("lifetime_value", pa.decimal128(16, 2)),
    pa.field("lifetime_orders", pa.int32()),
]
CUSTOMER_PROFILE_TYPE = pa.struct(CUSTOMER_PROFILE_FIELDS)
ORDER_LINE_FIELDS = [
    pa.field("product_sku", pa.string()),
    pa.field("category", pa.string()),
    pa.field("quantity", pa.int16()),
    pa.field("unit_price", pa.decimal128(16, 2)),
    pa.field("line_total", pa.decimal128(16, 2)),
]
ORDER_LINE_TYPE = pa.struct(ORDER_LINE_FIELDS)
ORDER_LINES_ARRAY_TYPE = pa.list_(ORDER_LINE_TYPE)
ATTRIBUTES_TYPE = pa.map_(pa.string(), pa.string())
GEO_FIELDS = [
    pa.field("city", pa.string()),
    pa.field("state", pa.string()),
    pa.field("postal_code", pa.string()),
    pa.field("latitude", pa.float64()),
    pa.field("longitude", pa.float64()),
]
GEO_TYPE = pa.struct(GEO_FIELDS)
EMAIL_LIST_TYPE = pa.list_(pa.string())


def decimal_from_cents(value: int) -> Decimal:
    return (Decimal(int(value)) / HUNDRED).quantize(Decimal("0.01"))


def cents_to_decimal_list(values: np.ndarray) -> list[Decimal]:
    return [decimal_from_cents(int(v)) for v in values]


def build_table(num_rows: int, seed: int = DEFAULT_SEED) -> pa.Table:
    rng = np.random.default_rng(seed)

    order_ids = np.arange(1, num_rows + 1, dtype=np.int64) + 1_000_000
    customer_ids = rng.integers(10_000, 999_999, size=num_rows, dtype=np.int64)

    first_name_idx = rng.integers(0, len(FIRST_NAMES), size=num_rows)
    last_name_idx = rng.integers(0, len(LAST_NAMES), size=num_rows)
    first_names = FIRST_NAMES[first_name_idx]
    last_names = LAST_NAMES[last_name_idx]

    loyalty_idx = rng.choice(len(LOYALTY_LEVELS), size=num_rows, p=LOYALTY_PROBS)
    loyalty_values = np.array(LOYALTY_LEVELS, dtype=object)[loyalty_idx]
    customer_segment_values = np.array([SEGMENT_BY_LOYALTY[level] for level in loyalty_values], dtype=object)
    public_mask = rng.random(num_rows) < 0.04
    customer_segment_values[public_mask] = "Public Sector"

    sales_channel_values = rng.choice(SALES_CHANNELS, size=num_rows)
    status_values = rng.choice(STATUS_OPTIONS, size=num_rows, p=STATUS_PROBS)
    payment_method_values = rng.choice(PAYMENT_METHODS, size=num_rows, p=PAYMENT_METHOD_PROBS)

    is_priority_mask = rng.random(num_rows) < 0.18

    ordered_quantity = rng.integers(1, 25, size=num_rows, dtype=np.int16)

    gross_weight = rng.normal(25.0, 7.5, size=num_rows)
    gross_weight = np.clip(gross_weight, 2.0, None)
    net_weight = gross_weight * rng.uniform(0.75, 0.95, size=num_rows)
    gross_weight = np.round(gross_weight, 3).astype(np.float32)
    net_weight = np.round(net_weight, 3).astype(np.float32)

    unit_price_cents = rng.integers(900, 56_000, size=num_rows, dtype=np.int64)
    discount_rates = rng.choice(DISCOUNT_RATES, size=num_rows, p=DISCOUNT_PROBS)
    gross_cents = ordered_quantity.astype(np.int64) * unit_price_cents
    discount_cents = np.rint(gross_cents * discount_rates).astype(np.int64)
    net_cents = gross_cents - discount_cents
    tax_rates = rng.choice(TAX_RATES, size=num_rows, p=TAX_RATE_PROBS)
    tax_cents = np.rint(net_cents * tax_rates).astype(np.int64)
    shipping_cents = rng.integers(0, 4_500, size=num_rows, dtype=np.int64)
    total_cents = net_cents + tax_cents + shipping_cents

    account_balance_cents = rng.integers(-250_000, 1_750_000, size=num_rows, dtype=np.int64)

    lifetime_orders = rng.integers(5, 900, size=num_rows, dtype=np.int32) + loyalty_idx * 25
    today = date(2025, 1, 1)
    signup_offsets = rng.integers(90, 3_650, size=num_rows, dtype=np.int32)
    signup_dates = [today - timedelta(days=int(offset)) for offset in signup_offsets]
    lifetime_value_cents = total_cents + rng.integers(50_000, 2_500_000, size=num_rows, dtype=np.int64) + loyalty_idx * 150_000

    base_order_date = date(2024, 1, 1)
    order_date_offsets = rng.integers(0, 365, size=num_rows, dtype=np.int32)
    ship_lag_days = rng.integers(1, 6, size=num_rows, dtype=np.int32)
    delivery_lag_days = rng.integers(2, 9, size=num_rows, dtype=np.int32)
    order_time_offsets_ms = rng.integers(0, 86_400_000, size=num_rows, dtype=np.int64)
    delivery_hour_offsets = rng.integers(0, 24, size=num_rows, dtype=np.int32)

    location_idx = rng.integers(0, len(LOCATIONS), size=num_rows)

    order_uuid_list: list[str] = []
    order_date_list: list[date] = []
    ship_date_list: list[date] = []
    order_ts_list: list[datetime] = []
    expected_delivery_ts_list: list[datetime] = []
    region_list: list[str] = []
    country_list: list[str] = []
    currency_list: list[str] = []
    geo_records: list[dict[str, object]] = []
    attributes_data: list[dict[str, str]] = []
    contact_emails: list[list[str]] = []
    order_lines_data: list[list[dict[str, object]]] = []

    # Build nested records row by row to keep related fields consistent.
    for i in range(num_rows):
        loc = LOCATIONS[location_idx[i]]
        region_list.append(loc["region"])
        country_list.append(loc["country"])
        currency_list.append(loc["currency"])
        geo_records.append(
            {
                "city": loc["city"],
                "state": loc["state"],
                "postal_code": loc["postal_code"],
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
            }
        )

        order_uuid_list.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, f"order-{order_ids[i]}")))

        order_date = base_order_date + timedelta(days=int(order_date_offsets[i]))
        order_date_list.append(order_date)
        ship_date_list.append(order_date + timedelta(days=int(ship_lag_days[i])))
        order_ts = datetime.combine(order_date, time.min, tzinfo=timezone.utc) + timedelta(
            milliseconds=int(order_time_offsets_ms[i])
        )
        order_ts_list.append(order_ts)
        expected_delivery_ts_list.append(
            order_ts + timedelta(days=int(delivery_lag_days[i]), hours=int(delivery_hour_offsets[i]))
        )

        first = str(first_names[i]).lower()
        last = str(last_names[i]).lower()
        primary_domain = str(rng.choice(DOMAINS))
        emails = [f"{first}.{last}@{primary_domain}"]
        if rng.random() < 0.35:
            alt_domain = str(rng.choice(DOMAINS))
            suffix = int(rng.integers(100, 999))
            emails.append(f"{first[0]}{last}{suffix}@{alt_domain}")
        contact_emails.append(emails)

        attributes_data.append(
            {
                "priority": "true" if bool(is_priority_mask[i]) else "false",
                "delivery_method": str(rng.choice(DELIVERY_METHODS)),
                "warehouse": loc["warehouse"],
                "order_type": str(rng.choice(ORDER_TYPES)),
                "region_code": loc["region_code"],
            }
        )

        line_count = int(rng.integers(1, 5))
        lines: list[dict[str, object]] = []
        for _ in range(line_count):
            product = PRODUCT_CATALOG[int(rng.integers(0, len(PRODUCT_CATALOG)))]
            line_qty = int(rng.integers(1, 8))
            price_cents = int(rng.integers(product["min_price"], product["max_price"]))
            line_total_cents = line_qty * price_cents
            lines.append(
                {
                    "product_sku": product["sku"],
                    "category": product["category"],
                    "quantity": line_qty,
                    "unit_price": decimal_from_cents(price_cents),
                    "line_total": decimal_from_cents(line_total_cents),
                }
            )
        order_lines_data.append(lines)

    customer_name_struct = pa.StructArray.from_arrays(
        [pa.array(first_names, type=pa.string()), pa.array(last_names, type=pa.string())],
        fields=CUSTOMER_NAME_FIELDS,
    )
    weights_struct = pa.StructArray.from_arrays(
        [pa.array(gross_weight, type=pa.float32()), pa.array(net_weight, type=pa.float32())],
        fields=WEIGHTS_FIELDS,
    )
    customer_profile_struct = pa.StructArray.from_arrays(
        [
            pa.array(loyalty_values, type=pa.string()),
            pa.array(signup_dates, type=pa.date32()),
            pa.array(cents_to_decimal_list(lifetime_value_cents), type=pa.decimal128(16, 2)),
            pa.array(lifetime_orders, type=pa.int32()),
        ],
        fields=CUSTOMER_PROFILE_FIELDS,
    )

    table = pa.table(
        {
            "order_id": pa.array(order_ids, type=pa.int64()),
            "order_uuid": pa.array(order_uuid_list, type=pa.string()),
            "customer_id": pa.array(customer_ids, type=pa.int64()),
            "customer_name": customer_name_struct,
            "customer_segment": pa.array(customer_segment_values, type=pa.string()),
            "account_balance": pa.array(cents_to_decimal_list(account_balance_cents), type=pa.decimal128(18, 2)),
            "region": pa.array(region_list, type=pa.string()),
            "country": pa.array(country_list, type=pa.string()),
            "sales_channel": pa.array(sales_channel_values, type=pa.string()),
            "status": pa.array(status_values, type=pa.string()),
            "order_date": pa.array(order_date_list, type=pa.date32()),
            "ship_date": pa.array(ship_date_list, type=pa.date32()),
            "order_ts": pa.array(order_ts_list, type=pa.timestamp("ms", tz="UTC")),
            "expected_delivery_ts": pa.array(expected_delivery_ts_list, type=pa.timestamp("ms", tz="UTC")),
            "is_priority": pa.array(is_priority_mask, type=pa.bool_()),
            "ordered_quantity": pa.array(ordered_quantity, type=pa.int16()),
            "weights_kg": weights_struct,
            "unit_price": pa.array(cents_to_decimal_list(unit_price_cents), type=pa.decimal128(14, 2)),
            "discount_rate": pa.array(discount_rates, type=pa.float32()),
            "subtotal": pa.array(cents_to_decimal_list(net_cents), type=pa.decimal128(18, 2)),
            "tax_amount": pa.array(cents_to_decimal_list(tax_cents), type=pa.decimal128(16, 2)),
            "shipping_cost": pa.array(cents_to_decimal_list(shipping_cents), type=pa.decimal128(14, 2)),
            "total_amount": pa.array(cents_to_decimal_list(total_cents), type=pa.decimal128(18, 2)),
            "currency": pa.array(currency_list, type=pa.string()),
            "payment_method": pa.array(payment_method_values, type=pa.string()),
            "customer_profile": customer_profile_struct,
            "contact_emails": pa.array(contact_emails, type=EMAIL_LIST_TYPE),
            "order_lines": pa.array(order_lines_data, type=ORDER_LINES_ARRAY_TYPE),
            "attributes": pa.array(attributes_data, type=ATTRIBUTES_TYPE),
            "geo": pa.array(geo_records, type=GEO_TYPE),
        }
    )

    table.validate(full=True)
    return table


def write_parquet(table: pa.Table, output_path: Path, *, row_groups: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dictionary_columns = [
        "region",
        "country",
        "customer_segment",
        "sales_channel",
        "status",
        "currency",
        "payment_method",
    ]

    writer_kwargs = {
        "use_dictionary": dictionary_columns,
        "compression": "zstd",
        "data_page_version": "2.0",
        "write_statistics": True,
    }

    try:
        writer = pq.ParquetWriter(str(output_path), table.schema, write_page_index=True, **writer_kwargs)
    except TypeError as exc:  # pragma: no cover - depends on pyarrow version
        raise RuntimeError(
            "The installed pyarrow version does not support page indexes. "
            "Upgrade to pyarrow >= 12.0 to regenerate the example dataset."
        ) from exc

    rows_per_group = max(math.ceil(table.num_rows / max(row_groups, 1)), 1)
    written_groups = 0
    try:
        for start in range(0, table.num_rows, rows_per_group):
            end = min(start + rows_per_group, table.num_rows)
            writer.write_table(table.slice(start, end - start))
            written_groups += 1
    finally:
        writer.close()

    return written_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the example Parquet dataset for parquet-analyzer.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROW_COUNT, help="Number of rows to generate (default: 100000).")
    parser.add_argument(
        "--row-groups",
        type=int,
        default=DEFAULT_ROW_GROUPS,
        help="Approximate number of row groups to write (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for deterministic dataset regeneration (default: 1337).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("example.parquet"),
        help="Target Parquet file path (default: examples/example.parquet).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = build_table(args.rows, seed=args.seed)
    row_group_count = write_parquet(table, args.output, row_groups=args.row_groups)
    print(
        f"Wrote {args.output} with {table.num_rows:,} rows, {table.num_columns} columns, "
        f"and {row_group_count} row groups."
    )


if __name__ == "__main__":
    main()
