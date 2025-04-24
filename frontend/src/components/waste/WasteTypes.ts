export interface WasteCategory {
  food_type: string;
  name: string;
  confidence: number;
  explanation: string;
}

export interface ClassificationResult {
  timestamp: string;
  image_id: string;
  contains_food: boolean;
  is_waste: boolean;
  categories: WasteCategory[];
}

/**
 * Attempts to create a valid ClassificationResult from the API response
 * This is a fallback for cases where the response format doesn't match our expectations
 */
export function createClassificationResult(data: any): ClassificationResult | null {
  // If data is already in the expected format, return it
  if (data && 
      typeof data === 'object' && 
      Array.isArray(data.categories) &&
      typeof data.timestamp === 'string') {
    return data as ClassificationResult;
  }
  
  // Try to extract information from the response
  try {
    // Default values
    let timestamp = new Date().toISOString();
    let image_id = "unknown.jpg";
    let contains_food = false;
    let is_waste = false;
    let categories: WasteCategory[] = [];
    
    if (typeof data === 'string') {
      // Try to extract timestamp
      const timestampMatch = data.match(/"timestamp":\s*"([^"]*)"/);
      if (timestampMatch && timestampMatch[1]) {
        timestamp = timestampMatch[1];
      }
      
      // Try to extract image_id
      const imageIdMatch = data.match(/"image_id":\s*"([^"]*)"/);
      if (imageIdMatch && imageIdMatch[1]) {
        image_id = imageIdMatch[1];
      }
      
      // Try to extract contains_food and is_waste
      const containsFoodMatch = data.match(/"contains_food":\s*(true|false)/);
      if (containsFoodMatch && containsFoodMatch[1]) {
        contains_food = containsFoodMatch[1] === 'true';
      }
      
      const isWasteMatch = data.match(/"is_waste":\s*(true|false)/);
      if (isWasteMatch && isWasteMatch[1]) {
        is_waste = isWasteMatch[1] === 'true';
      }
      
      // Try to extract categories
      // Look for food_type, name, confidence, and explanation
      const categoryMatches = [...data.matchAll(/"food_type":\s*"([^"]*)".+?"name":\s*"([^"]*)".+?"confidence":\s*([\d.]+).+?"explanation":\s*"([^"]*)"/g)];
      
      if (categoryMatches && categoryMatches.length > 0) {
        categories = categoryMatches.map(match => ({
          food_type: match[1] || "",
          name: match[2] || "",
          confidence: parseFloat(match[3] || "0"),
          explanation: match[4] || ""
        }));
      }
    }
    
    // If we managed to extract at least one category, consider this a success
    if (categories.length > 0) {
      return {
        timestamp,
        image_id,
        contains_food: true, // If we found categories, we have food
        is_waste: true,      // If we found categories, it's waste
        categories
      };
    }
    
    return null;
  } catch (error) {
    console.error('Error creating classification result:', error);
    return null;
  }
}

/**
 * Parse a specific response format that includes a descriptive text followed by JSON data
 * Example: "The image shows a pile of pumpkins... *Answer*: {"timestamp": "2025-04-13T11:02:49.175739+00:00", ...}"
 */
export function parseResponseWithDescription(response: string): ClassificationResult | null {
  // Look for the pattern with "*Answer*: {JSON}"
  const answerMatch = response.match(/\*Answer\*:\s*(\{.*\})/s);
  if (answerMatch && answerMatch[1]) {
    try {
      const resultJson = JSON.parse(answerMatch[1]);
      return resultJson as ClassificationResult;
    } catch (error) {
      console.error('Error parsing answer section:', error);
    }
  }
  
  // Another common pattern is where a description is followed by JSON without specific marker
  const jsonMatch = response.match(/.*?((\{.*\})\s*$)/s);
  if (jsonMatch && jsonMatch[2]) {
    try {
      const resultJson = JSON.parse(jsonMatch[2]);
      return resultJson as ClassificationResult;
    } catch (error) {
      console.error('Error parsing trailing JSON:', error);
    }
  }
  
  return null;
}

/**
 * Extract just the essential waste classification data, even if the full JSON can't be parsed
 * This is for fallback scenarios where we need to at least show the core information
 */
export function extractEssentialWasteData(response: string): WasteCategory[] | null {
  try {
    const categories: WasteCategory[] = [];
    
    // Pattern to match food_type, name and explanation fields with their values
    const pattern = /"food_type"\s*:\s*"([^"]*)"\s*,.*?"name"\s*:\s*"([^"]*)"\s*,.*?"confidence"\s*:\s*([\d.]+).*?"explanation"\s*:\s*"([^"]*)"/gs;
    
    // Find all matches
    const matches = [...response.matchAll(pattern)];
    
    if (matches.length > 0) {
      for (const match of matches) {
        const [_, food_type, name, confidence, explanation] = match;
        categories.push({
          food_type: food_type || "unknown",
          name: name || "unknown",
          confidence: parseFloat(confidence || "0"),
          explanation: explanation || ""
        });
      }
      return categories;
    }
    
    // Fallback to more basic pattern matching if the above fails
    // Look for any quoted strings that might be food_type, name or explanation
    if (response.includes("food_type") && response.includes("name") && response.includes("explanation")) {
      // Try to extract the food type
      const foodTypeMatch = response.match(/"food_type"\s*:\s*"([^"]*)"/);
      const nameMatch = response.match(/"name"\s*:\s*"([^"]*)"/);
      const explanationMatch = response.match(/"explanation"\s*:\s*"([^"]*)"/);
      
      if (foodTypeMatch && nameMatch && explanationMatch) {
        categories.push({
          food_type: foodTypeMatch[1] || "unknown",
          name: nameMatch[1] || "unknown",
          confidence: 0.8, // Default confidence if not found
          explanation: explanationMatch[1] || ""
        });
        return categories;
      }
    }
    
    return null;
  } catch (error) {
    console.error("Error extracting essential waste data:", error);
    return null;
  }
} 