-- 1. Rata-rata umur customer jika dilihat dari marital statusnya 
select "Marital Status", round(avg(age)) average_age
from "Case Study - Customer.csv" cscc 
group by "Marital Status" 

-- 2. Rata-rata umur customer jika dilihat dari gendernya
select gender , round(avg(age)) average_age
from "Case Study - Customer.csv" cscc
group by gender  

-- 3. Nama store dengan total quantity terbanyak
select sum(qty) total_qty,cssc.storename 
from "Case Study - Store.csv" cssc 
join "Case Study - Transaction.csv" cstc 
on cssc.storeid = cstc.storeid 
group by cssc.storename 
order by 1 desc 
limit 1

-- 4. Nama produk terlaris dengan total amount terbanyak
select sum(totalamount) total_amount,  cspc."Product Name" 
from "Case Study - Product.csv" cspc 
join "Case Study - Transaction.csv" cstc2 
on cspc.productid = cstc2.productid 
group by cspc."Product Name" 
order by 1 desc
limit 1


