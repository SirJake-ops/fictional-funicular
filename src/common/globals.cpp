//
// Created by jake on 1/13/26.
//

enum class REST { GET, POST, PUT, PATCH, DELETE };
const char *to_string(const REST e) {
    switch (e) {
        case REST::GET:
            return "GET";
        case REST::POST:
            return "POST";
        case REST::PUT:
            return "PUT";
        case REST::PATCH:
            return "PATCH";
        case REST::DELETE:
            return "DELETE";
    }

    return "UNKNOWN";
}
