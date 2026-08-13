-- ============================================================
-- 数据库 DDL —— 四张业务表 + 两张聚合视图
-- 数据库名: ai_analytics
-- ============================================================

CREATE DATABASE IF NOT EXISTS ai_analytics
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE ai_analytics;

-- -----------------------------------------------------------
-- 1. customers（清洗后的统一客户表，ETL导入）
-- -----------------------------------------------------------
DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
    customer_id             VARCHAR(20)     PRIMARY KEY,
    country                 VARCHAR(100)    NOT NULL,
    age                     INT             NOT NULL,
    gender                  VARCHAR(10)     NOT NULL,
    membership_tier         VARCHAR(20)     NOT NULL,
    registration_date       DATE            NOT NULL,
    total_orders            INT             NOT NULL DEFAULT 0,
    total_spend_usd         DECIMAL(12, 2)  NOT NULL DEFAULT 0.00,
    avg_order_value_usd     DECIMAL(10, 2)  NOT NULL DEFAULT 0.00,
    days_since_last_purchase INT            NOT NULL DEFAULT 0,
    preferred_category      VARCHAR(100)    NOT NULL,
    preferred_device        VARCHAR(50)     NOT NULL,
    preferred_payment_method VARCHAR(100)   NOT NULL,
    acquisition_channel     VARCHAR(100)    NOT NULL,
    reviews_given           INT             NOT NULL DEFAULT 0,
    avg_review_score        DECIMAL(3, 1)   NOT NULL DEFAULT 0.0,
    returns_made            INT             NOT NULL DEFAULT 0,
    wishlist_items          INT             NOT NULL DEFAULT 0,
    newsletter_subscribed   TINYINT         NOT NULL DEFAULT 0,
    churned                 TINYINT         NOT NULL DEFAULT 0,

    INDEX idx_country (country),
    INDEX idx_membership (membership_tier),
    INDEX idx_churned (churned),
    INDEX idx_total_spend (total_spend_usd)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- -----------------------------------------------------------
-- 2. orders（订单明细表，原样导入，customer_rating保留NULL）
-- -----------------------------------------------------------
DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
    order_id                    VARCHAR(20)     PRIMARY KEY,
    customer_id                 VARCHAR(20)     NOT NULL,
    order_date                  DATE            NOT NULL,
    year                        INT             NOT NULL,
    month                       INT             NOT NULL,
    quarter                     INT             NOT NULL,
    day_of_week                 VARCHAR(10)     NOT NULL,
    product_name                VARCHAR(200)    NOT NULL,
    category                    VARCHAR(100)    NOT NULL,
    unit_price_usd              DECIMAL(10, 2)  NOT NULL,
    quantity                    INT             NOT NULL,
    subtotal_usd                DECIMAL(10, 2)  NOT NULL,
    discount_pct                DECIMAL(5, 2)   NOT NULL DEFAULT 0.00,
    discount_amount_usd         DECIMAL(10, 2)  NOT NULL DEFAULT 0.00,
    shipping_fee_usd            DECIMAL(8, 2)   NOT NULL DEFAULT 0.00,
    tax_pct                     DECIMAL(5, 2)   NOT NULL DEFAULT 0.00,
    tax_amount_usd              DECIMAL(10, 2)  NOT NULL DEFAULT 0.00,
    total_amount_usd            DECIMAL(10, 2)  NOT NULL,
    payment_method              VARCHAR(50)     NOT NULL,
    device_used                 VARCHAR(50)     NOT NULL,
    delivery_days               INT             NOT NULL,
    delivery_date               DATE            NOT NULL,
    order_status                VARCHAR(30)     NOT NULL,
    returned                    TINYINT         NOT NULL DEFAULT 0,
    customer_rating             DECIMAL(3, 1)   NULL COMMENT '天然缺失约63%，保留NULL供数据治理Agent评分',
    session_duration_minutes    DECIMAL(8, 2)   NOT NULL DEFAULT 0.00,
    pages_viewed_before_purchase INT            NOT NULL DEFAULT 0,
    is_repeat_customer          TINYINT         NOT NULL DEFAULT 0,

    INDEX idx_customer (customer_id),
    INDEX idx_order_date (order_date),
    INDEX idx_year_month (year, month),
    INDEX idx_category (category),
    INDEX idx_returned (returned)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- -----------------------------------------------------------
-- 3. monthly_revenue（月粒度营收聚合表，75行，原样导入）
-- -----------------------------------------------------------
DROP TABLE IF EXISTS monthly_revenue;
CREATE TABLE monthly_revenue (
    year                INT             NOT NULL,
    month               INT             NOT NULL,
    quarter             INT             NOT NULL,
    orders              INT             NOT NULL,
    revenue_usd         DECIMAL(14, 2)  NOT NULL,
    avg_order_value     DECIMAL(10, 2)  NOT NULL,
    avg_discount_pct    DECIMAL(5, 2)   NOT NULL,
    return_rate         DECIMAL(5, 4)   NOT NULL,
    unique_customers    INT             NOT NULL,
    new_customers       INT             NOT NULL,

    PRIMARY KEY (year, month)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- -----------------------------------------------------------
-- 4. product_summary（商品维度汇总表，140行，原样导入）
-- -----------------------------------------------------------
DROP TABLE IF EXISTS product_summary;
CREATE TABLE product_summary (
    category            VARCHAR(100)    NOT NULL,
    product_name        VARCHAR(200)    PRIMARY KEY,
    total_orders        INT             NOT NULL,
    total_revenue_usd   DECIMAL(14, 2)  NOT NULL,
    avg_price           DECIMAL(10, 2)  NOT NULL,
    avg_rating          DECIMAL(3, 1)   NOT NULL,
    return_rate         DECIMAL(5, 4)   NOT NULL,
    avg_discount_pct    DECIMAL(5, 2)   NOT NULL,
    avg_delivery_days   DECIMAL(5, 2)   NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- -----------------------------------------------------------
-- 5. 日粒度聚合视图（供 Prediction Agent 查询）
-- -----------------------------------------------------------
CREATE OR REPLACE VIEW daily_orders_agg AS
SELECT
    order_date,
    year,
    month,
    COUNT(*)                                AS order_count,
    SUM(subtotal_usd)                       AS revenue_usd,
    AVG(subtotal_usd)                       AS avg_order_value,
    AVG(discount_pct)                       AS avg_discount_pct,
    COUNT(DISTINCT customer_id)             AS unique_customers,
    SUM(CASE WHEN is_repeat_customer = 1 THEN 1 ELSE 0 END) AS repeat_orders,
    AVG(delivery_days)                      AS avg_delivery_days,
    AVG(session_duration_minutes)           AS avg_session_minutes,
    AVG(pages_viewed_before_purchase)       AS avg_pages_viewed,
    AVG(customer_rating)                    AS avg_customer_rating,
    COUNT(CASE WHEN returned = 1 THEN 1 END) AS return_count
FROM orders
GROUP BY order_date, year, month
ORDER BY order_date;


-- -----------------------------------------------------------
-- 6. 月粒度聚合视图（供 Prophet 预测训练使用）
-- -----------------------------------------------------------
CREATE OR REPLACE VIEW monthly_orders_agg AS
SELECT
    year,
    month,
    COUNT(*)                                AS order_count,
    SUM(subtotal_usd)                       AS revenue_usd,
    AVG(subtotal_usd)                       AS avg_order_value,
    AVG(discount_pct)                       AS avg_discount_pct,
    COUNT(DISTINCT customer_id)             AS unique_customers,
    SUM(CASE WHEN is_repeat_customer = 1 THEN 1 ELSE 0 END) AS repeat_orders,
    AVG(customer_rating)                    AS avg_customer_rating,
    COUNT(CASE WHEN returned = 1 THEN 1 END) AS return_count
FROM orders
GROUP BY year, month
ORDER BY year, month;
