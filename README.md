# Modelling Airbnb's Property Listing Dataset

Creating a multimodal deep learning structure that processes text, images and tabular data to predict the type of property from an Airbnb listing.


## Data preparation

The 'structured' data is downloaded as a `.csv` file from an AWS S3 bucket containing the following information:
- ID: Unique identifier for the listing
- Category: The category of the listing
- Title: The title of the listing
- Description: The description of the listing
- Amenities: The available amenities of the listing
- Location: The location of the listing
- guests: The number of guests that can be accommodated in the listing
- beds: The number of available beds in the listing
- bathrooms: The number of bathrooms in the listing
- Price_Night: The price per night of the listing
- Cleanliness_rate: The cleanliness rating of the listing
- Accuracy_rate: How accurate the description of the listing is, as reported by previous guests
 -Location_rate: The rating of the location of the listing
- Check-in_rate: The rating of check-in process given by the host
- Value_rate: The rating of value given by the host
- amenities_count: The number of amenities in the listing
- url: The URL of the listing
- bedrooms: The number of bedrooms in the listing

The file `tabular_data.py` contains the function `clean_data()` which will take a DataFrame as an input, and return a cleaned DataFrame. This is saved in the `dataframes` directory as `cleaned_dataset.csv`. A Tableau dashboard is created to take an overview at the data provided.

<p align='center'>
  <img src='README-images/airbnb-listing-dashboard.png' width='750'>
</p>

> A Tableau dashboard showing initial insights into the data.
