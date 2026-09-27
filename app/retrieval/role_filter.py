def filter_by_role(results, user_role):
    """
    Keep only documents that the user is authorized to access.
    """

    filtered_results = []

    for result in results:

        document = result["document"]

        allowed_roles = document.get(
            "allowed_roles",
            []
        )

        if user_role in allowed_roles:

            filtered_results.append(result)

    return filtered_results