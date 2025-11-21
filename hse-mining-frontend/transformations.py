# This module contains functions that are intended to process the data received from LLM API in-place, 
# e.g. to prepare the data to be better suited for further pipeline steps. This module is reserved for
# in-place transformations only, and should not contain any functions that interact directly with the
# LLM API or any backend system

def transform_observation(observations):
 
    columns = ', '.join(observations.columns)
    rows = str(len(observations))
    observations_sorted = observations.to_dict(orient='records')
   
    observations_sorted = '; '.join([', '.join(
        [str(value) for value in record.values()]) for record in observations_sorted])
    return observations_sorted, columns, rows