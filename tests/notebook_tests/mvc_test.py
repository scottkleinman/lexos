# API Module
"""This script is a generic template for testing API feedback to an MVC framework.

It performs API functions and raises a LexosException if they fail.

Important: The error message will be displayed if the function is called
directly, but not if it is called from a function in the Controller module.
This allows the controller module to handle an error message without exiting
the program.
"""

from typing import Any

# Import API modules
# from lexos.io.smart import Load

# Note look for "api_method" in the code below and change it to the method you wish to test.

# LexosException Module
class LexosException(Exception):
    """Generic LexosException class."""

    pass


# Model Module
class TemplateModel:
    """Replace methods as necessary from the API module being tested ."""

    pass

    def get_api_feedback(self, input: Any) -> Any:
        """Call the API to read feedback into the model."""
        # Feed the input to the API
        model_value = api_method(input)
        # Return the model value
        return model_value


# View Module
class View:
    """View class."""

    def __init__(self, show_api_errors: bool = True) -> None:
        """Initialize the view.

        Args:
            show_api_errors (bool): Whether to show API errors.

        Returns:
            None
        """
        self.show_api_errors = show_api_errors

    def display(self, model_value: Any) -> None:
        """Display a value in the model.

        Args:
            model_value (str): The value to show.

        Returns:
            None
        """
        print(f"Good news! The model returned {model_value}.")

    def display_custom_error(self, item: Any, err: Any = None) -> None:
        """Display a custom error with or without Python error messages.

        Args:
            item (Any): The item that caused the error.
            err (Any): The error message.

        Returns:
            None
        """
        print(f"{item} caused this custom error.")
        if err and self.show_api_errors:
            print(f"Python raised this error: {err}")

    def display_python_error(self, err: str) -> None:
        """Display a Python error returned by LexosException.

        Args:
            err (str): The error message.

        Returns:
            None
        """
        print(err)


# Controller Module
class Controller:
    """Controller class."""

    def __init__(self, model: object, view: object) -> None:
        """Constructor for the Controller class."""
        self.model = model
        self.view = view

    def display_item(self, item_name: str) -> None:
        """Provide the model with input and what API method to call."""
        # If the call is successful, show the item.
        try:
            # Call the API to read an item from the model
            feedback = self.model.api_method(item_name)
            # Call the View to show the item
            self.view.display(feedback)
        # Otherwise, display the error message.
        except LexosException as e:
            self.view.display_custom_error(item_name, e)
            # self.view.display_error(e)


# Main Module
def main():
    """Main function."""

    # Data
    my_items = []

    # Instantiate a controller - the boolean is for showing API errors
    c = Controller(TemplateModel(), View(show_api_errors=True))

    # Iterate through the data and call controller methods
    for item in my_items:
        c.display_item(item)


if __name__ == "__main__":
    main()
