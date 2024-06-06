class Company {
  // Private property
  #revenue;

  constructor(name, revenue) {
    this.name = name;
    this.#revenue = revenue;
  }

  // Getter for private property
  get revenue() {
    return this.#revenue;
  }

  // Setter for private property
  set revenue(value) {
    if (value >= 0) {
      this.#revenue = value;
    } else {
      console.error("Revenue can't be negative");
    }
  }

  // Method
  displayInfo() {
    console.log(`Company Name: ${this.name}`);
    console.log(`Revenue: ${this.#revenue}`);
  }
}

class TechCompany extends Company {
  constructor(name, revenue, techFocus) {
    super(name, revenue);
    this.techFocus = techFocus;
  }

  // Overriding displayInfo method (Polymorphism)
  displayInfo() {
    super.displayInfo();
    console.log(`Tech Focus: ${this.techFocus}`);
  }
}
// Create instances of TechCompany and ManufacturingCompany
const google = new TechCompany('Google', 182500, 'Internet Services');

// Use the setter to update revenue
google.revenue = 200000;

// Display information
google.displayInfo();