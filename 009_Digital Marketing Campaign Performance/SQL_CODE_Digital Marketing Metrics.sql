create database marketing;
USE marketing;

create table campaign_data (
	id int,
    c_date date,
    campaign_name varchar (300),
    category varchar (150),
    campaign_id bigint,
    impressions int,
    mark_spent float,
    clicks int,
    leads int,
    orders int,
    revenue float
    );
    
select * from campaign_data;
select count(*) from campaign_data;

-- Marketing KPIs --

-- Return on Marketing Investment (ROMI) --

select 
	revenue, 
    mark_spent,
    round((revenue-mark_spent)/mark_spent, 2) as romi
from campaign_data
where impressions > 0 and clicks > 0 and leads > 0 and orders > 0;
	
-- Click-Through Rate (CTR) --

select
	clicks,
    impressions,
    round(clicks / impressions, 4) as ctr
from campaign_data
where impressions > 0 and clicks > 0 and leads > 0 and orders > 0;

-- Conversion Rate 1 (Clicks → Leads) ---

select
	leads,
    clicks,
    round(leads/clicks, 4) as conversion_rate_1
from campaign_data
where impressions > 0 and clicks > 0 and leads > 0 and orders > 0;

-- Conversion Rate 2 (Leads → Orders) --

select
	leads,
    orders,
    round(orders/leads, 4) as conversion_rate_2
from campaign_data
where impressions > 0 and clicks > 0 and leads > 0 and orders > 0;

-- Average Order Value (AOV) --

select
	revenue,
    orders,
    round(revenue/orders, 2) as aov
from campaign_data
where impressions > 0 and clicks > 0 and leads > 0 and orders > 0;

-- Cost Per Click (CPC) --

select
	mark_spent,
    clicks,
    round(mark_spent/clicks, 2) as cpc
from campaign_data
where impressions > 0 and clicks > 0 and leads > 0 and orders > 0;

-- Cost Per Lead (CPL) --

select
	mark_spent,
    leads,
    round(mark_spent/leads, 2) as cpl
from campaign_data
where impressions > 0 and clicks > 0 and leads > 0 and orders > 0;   

-- Customer Acquisition Cost (CAC) --

select
	mark_spent,
    orders,
    round(mark_spent/orders, 2) as cac
from campaign_data
where impressions > 0 and clicks > 0 and leads > 0 and orders > 0; 


-- Top10 Campaigns in ROMI --

select
	campaign_name,
    round(AVG((revenue-mark_spent)/mark_spent), 2) as avg_romi,
    SUM(mark_spent) as total_spent,
    sum(revenue) as total_revenue
from campaign_data
where mark_spent > 0
group by campaign_name
order by avg_romi desc
limit 10;

-- Top10 in Revenue --

select
	c_date,
    sum(revenue) as total_revenue,
    sum(mark_spent) as total_spent
from campaign_data
group by c_date
order by total_revenue desc
limit 10;

-- AVG ROMI per Category --

select
	category,
    round(AVG((revenue-mark_spent)/mark_spent), 2) as avg_romi,
	sum(revenue) as total_revenue,
    sum(mark_spent) as mark_spent
from campaign_data
where mark_spent > 0
group by category
order by avg_romi desc;

-- AVG CPC, CPL, CAC per category --

select 
  category,
  round(sum(mark_spent)/sum(clicks), 2) AS avg_cpc,
  round(sum(mark_spent)/sum(leads), 2) AS avg_cpl,
  round(sum(mark_spent)/sum(orders), 2) AS avg_cac
from campaign_data
where clicks > 0 and leads > 0 and orders > 0
group by category
order by avg_cac asc;

-- Orders per Week Day --

select
	dayname(c_date) as day_name,
    count(*) as campaigns_run,
    round(sum(revenue), 2) as total_revenue,
	round(sum(orders), 2) as total_orders
from campaign_data
group by dayname(c_date)
order by total_revenue desc;    
    
-- Daily Revenue & Spend --

select
	c_date,
    sum(revenue) as total_revenue,
    sum(mark_spent) as total_spent
from campaign_data
group by c_date
order by c_date asc;














